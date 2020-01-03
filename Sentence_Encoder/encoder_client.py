"""A client for running inference with an Encoder model.

Copyright PolyAI Limited.
"""
import os
import logging
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.CRITICAL)
import time
from collections import OrderedDict
from functools import wraps
from threading import Lock

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub
import tensorflow_text

[tensorflow_text]


class EncoderClient:
    """A client for running inference with a ConveRT encoder model.

    This wraps tensorflow hub, and gives an interface to input text, and
    get numpy encoding vectors in return. It includes a few optimizations to
    make encoding faster: deduplication of inputs, caching, and internal
    batching.

    You can pass as many input sentences as you like to `encode_sentence`,
    `encode_contexts` and `encode_responses`, and internally they will be
    batched to prevent going out of memory.

    Args:
        uri: the tensorflow hub URI of the model to load.
        use_extra_context: whether the model uses extra context features.
        use_extra_context_prefixes: whether to add 0:, 1: etc. as prefixes to
            the extra context features. The reddit model is not trained like
            this, but the ubuntu model is fine-tuned like this.
        max_extra_contexts: the maximum number of extra contexts to pass to
            the model.
        cache_size: the number of encodings for each function to cache in
            memory. 0 to disable.
        internal_batch_size: the batch size to use internally.
    """

    def __init__(
            self,
            uri,
            use_extra_context=False,
            use_extra_context_prefixes=False,
            max_extra_contexts=10,
            cache_size=65_536,
            internal_batch_size=64,
    ):
        self._use_extra_context = use_extra_context
        self._use_extra_context_prefixes = use_extra_context_prefixes
        self._max_extra_contexts = max_extra_contexts

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        self._session = tf.Session(graph=tf.Graph(), config=config)
        self._internal_batch_size = internal_batch_size
        with self._session.graph.as_default():
            embed_fn = tensorflow_hub.Module(uri)

            self._fed_contexts = tf.placeholder(
                shape=[None], dtype=tf.string)
            self._fed_extra_contexts = tf.placeholder(
                shape=[None], dtype=tf.string)
            self._fed_responses = tf.placeholder(
                shape=[None], dtype=tf.string)

            if use_extra_context:
                self._context_embeddings = embed_fn(
                    {
                        'context': self._fed_contexts,
                        'extra_context': self._fed_extra_contexts,
                    },
                    signature="encode_context",
                )
            else:
                self._context_embeddings = embed_fn(
                    self._fed_contexts,
                    signature="encode_context",
                )

            self._response_embeddings = embed_fn(
                self._fed_responses, signature="encode_response",
            )
            self._sentence_embeddings = embed_fn(self._fed_contexts)
            init_ops = (
                tf.global_variables_initializer(), tf.tables_initializer())

        self._session.run(init_ops)

        if cache_size > 0:
            self.encode_sentences = cache_encodings(
                self.encode_sentences,
                cache_size=cache_size,
            )
            self._encode_contexts_cacheable = cache_encodings(
                self._encode_contexts_cacheable,
                cache_size=cache_size,
            )
            self.encode_responses = cache_encodings(
                self.encode_responses,
                cache_size=cache_size,
            )

    def encode_sentences(self, sentences):
        """Encode the given texts with the general sentence encoder."""
        return _batch_session_run(
            self._sentence_embeddings, {self._fed_contexts: sentences},
            self._session, self._internal_batch_size, self._fed_contexts,
        )

    def encode_contexts(self, contexts, extra_contexts=None):
        """Encode the given texts as contexts.

        Args:
            contexts: the most recent contexts.
            extra_contexts: a list of lists, containing the previous contexts
                for each example. These are ordered from oldest to most recent.
        """
        if not self._use_extra_context:
            if extra_contexts is not None:
                raise ValueError(
                    "This encoder does not use extra contexts, but extra "
                    "contexts were passed.")
            extra_contexts = [[] for _ in contexts]
        elif extra_contexts is None:
            extra_contexts = [[] for _ in contexts]

        extra_context_features = []
        for extra_context in extra_contexts:
            extra_context = list(extra_context)
            extra_context.reverse()
            extra_context = extra_context[:self._max_extra_contexts]
            if self._use_extra_context_prefixes:
                extra_context = [
                    "{}: {}".format(i, text)
                    for i, text in enumerate(extra_context)
                ]
            extra_context_features.append(" ".join(extra_context))

        examples = list(zip(contexts, extra_context_features))
        return self._encode_contexts_cacheable(examples)

    def _encode_contexts_cacheable(self, examples):
        """Encode the given texts as contexts, with a cacheable signature."""
        contexts, extra_contexts = zip(*examples)
        feed_dict = {self._fed_contexts: contexts}
        feature_for_length = self._fed_contexts

        if self._use_extra_context:
            feed_dict[self._fed_extra_contexts] = extra_contexts
            feature_for_length = self._fed_extra_contexts

        return _batch_session_run(
            self._context_embeddings, feed_dict,
            self._session, self._internal_batch_size, feature_for_length,
        )

    def encode_responses(self, responses):
        """Encode the given texts as responses."""
        return _batch_session_run(
            self._response_embeddings, {self._fed_responses: responses},
            self._session, self._internal_batch_size, self._fed_responses,
        )


_LOG_EVERY_SECS = 20


def _batch_session_run(
    output_tensor, feed_dict, session, batch_size, feature_for_length,
):
    """Evaluates the output_tensors given the feeds in batches."""
    batch_dims = [len(v) for v in feed_dict.values()]
    assert all([batch_dim == batch_dims[0] for batch_dim in batch_dims]), (
        "_batch_session_run requires fed values be the same batch dimension. "
        f"inputs {list(feed_dict.values())}"
    )
    total_size = batch_dims[0]

    # Sort examples by the estimated length of the `feature_for_length` feature
    # so they are batched together in batches of similar length. This improves
    # encoding speed, as batches with small sequences run much faster.
    feature_lengths = [
        _estimate_num_tokens(sentence)
        for sentence in feed_dict[feature_for_length]
    ]
    ordering = np.argsort(feature_lengths)
    ordering_rev = np.argsort(ordering)
    feed_dict = {
        key: [value[i] for i in ordering]
        for key, value in feed_dict.items()
    }

    outputs = []
    #start_time = time.time()
    #last_log = start_time
    #glog.info(f"Encoding {total_size} examples.")
    for i in range(0, total_size, batch_size):

        """

        if (time.time() - last_log) > _LOG_EVERY_SECS:
            glog.info(
                f"Encoded {i} / {total_size}  ({i/total_size:.1%})")
            last_log = time.time()
        """

        batch_feed_dict = {
            k: np.array(v[i:i + batch_size]) for k, v in feed_dict.items()
        }
        outputs.append(
            session.run(output_tensor, feed_dict=batch_feed_dict))
    """
    total_time = time.time() - start_time
    glog.info(
        "Encoded %i examples in %.3f seconds",
        total_size, total_time,
    )
    """
    return np.concatenate(outputs)[ordering_rev]


def _estimate_num_tokens(sentence):
    """Estimates the number of tokens a sentence may have."""
    return len(sentence.split())


def cache_encodings(encoding_function, cache_size):
    """A decorator that allows caching an encoding function.

    This is similar to `functools.lru_cache`, except it works on an encoding
    function that takes batches of examples, producing a vector encoding for
    each. It caches the encoding for each individual example in a batch, and
    deals with composing the final numpy matrix. (Batching is important for
    neural network computational efficiency.)

    Args:
        encoding_function: a function that takes a list of examples and returns
            a numpy matrix of encodings, whose number of rows is equal to the
            length of the list. Examples can be anything hashable, but
            typically either strings or lists of strings (e.g. extra contexts).
            Lists are converted to tuples internally to make them hashable.
        cache_size: the maximum number of encodings to cache. Defaults to
            65 536 = 2^16, performs best if set to a power of 2.

    Returns:
        a cached version of `encoding_function`.
        This has the following attributes:
            cache_hits: the number of times the cache has been hit.

    """
    cache = OrderedDict()
    # Protect modification of the cache with a lock.
    lock = Lock()
    # Number of cache hits.
    hits = 0

    @wraps(encoding_function)
    def _cached_function(examples):
        nonlocal hits
        # Ensure the input is hashable.
        examples = _convert_lists_to_tuples(examples)
        unique_examples = set(examples)
        encodings = {}
        uncached_examples = []
        for example in unique_examples:
            with lock:
                value = cache.get(example)
                if value is not None:
                    hits += 1
                    encodings[example] = value
                    # Move the key to the front of the cache.
                    cache.move_to_end(example)
                else:
                    uncached_examples.append(example)

        if uncached_examples:
            # Compute the encodings of anything not found in the cache.
            uncached_encodings = encoding_function(uncached_examples)
            for example, encoding in zip(
                    uncached_examples, uncached_encodings):
                encodings[example] = encoding
                with lock:
                    cache[example] = encoding

        with lock:
            while len(cache) > cache_size:
                # Remove the least-recently used items.
                cache.popitem(last=False)

        # Construct the encoding matrix.
        return np.array([
            encodings.get(example) for example in examples
        ])

    def cache_hits():
        return hits

    _cached_function.cache_hits = cache_hits

    return _cached_function


def _convert_lists_to_tuples(x):
    if isinstance(x, (list, tuple)):
        return tuple(_convert_lists_to_tuples(item) for item in x)
    return x
