# coding=utf-8
# Copyright 2019 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""This module implements a simple parser that can be used for TAPAS.

Given a table, a question and one or more answer_texts, it will parse the texts
to populate other fields (e.g. answer_coordinates, float_value) that are required
by TAPAS.

Please note that exceptions in this module are concise and not parameterized,
since they are used as counter names in a BEAM pipeline.
"""

import enum
from typing import Callable, List, Text, Optional
import collections
import six
import struct
import unicodedata
import re

import frozendict
import numpy as np
import scipy.optimize
#from lapsolver import solve_dense

class SupervisionMode(enum.Enum):
    # Don't filter out any supervised information.
    NONE = 0
    # Remove all the supervised signals and recompute them by parsing answer
    # texts.
    REMOVE_ALL = 2
    # Same as above but discard ambiguous examples
    # (where an answer matches multiple cells).
    REMOVE_ALL_STRICT = 3


def _find_matching_coordinates(table, answer_text,
                               normalize):
    normalized_text = normalize(answer_text)
    for row_index, row in table.iterrows():
        for column_index, cell in enumerate(row):
            if normalized_text == normalize(str(cell)):
                yield (row_index, column_index)


def _compute_cost_matrix_inner(
        table,
        answer_texts,
        normalize,
        discard_ambiguous_examples,
):
    """Returns a cost matrix M where the value M[i,j] contains a matching cost from answer i to cell j.

    The matrix is a binary matrix and -1 is used to indicate a possible match from
    a given answer_texts to a specific cell table. The cost matrix can then be
    usedto compute the optimal assignments that minimizes the cost using the
    hungarian algorithm (see scipy.optimize.linear_sum_assignment).

    Args:
      table: a Pandas dataframe.
      answer_texts: a list of strings.
      normalize: a function that normalizes a string.
      discard_ambiguous_examples: If true discard if answer has multiple matches.

    Raises:
      ValueError if:
        - we cannot correctly construct the cost matrix or the text-cell
        assignment is ambiguous.
        - we cannot find a matching cell for a given answer_text.

    Returns:
      A numpy matrix with shape (num_answer_texts, num_rows * num_columns).
    """
    max_candidates = 0
    n_rows, n_columns = table.shape[0], table.shape[1]
    num_cells = n_rows * n_columns
    num_candidates = np.zeros((n_rows, n_columns))
    cost_matrix = np.zeros((len(answer_texts), num_cells))

    for index, answer_text in enumerate(answer_texts):
        found = 0
        for row, column in _find_matching_coordinates(table, answer_text,
                                                      normalize):
            found += 1
            cost_matrix[index, (row * len(table.columns)) + column] = -1
            num_candidates[row, column] += 1
            max_candidates = max(max_candidates, num_candidates[row, column])
        if found == 0:
            return None
        if discard_ambiguous_examples and found > 1:
            raise ValueError("Found multiple cells for answers")

    # TODO(piccinno): Shall we allow ambiguous assignments?
    if max_candidates > 1:
        raise ValueError("Assignment is ambiguous")

    return cost_matrix


def _compute_cost_matrix(
        table,
        answer_texts,
        discard_ambiguous_examples,
):
    """Computes cost matrix."""
    for index, normalize_fn in enumerate(STRING_NORMALIZATIONS):
        try:
            result = _compute_cost_matrix_inner(
                table,
                answer_texts,
                normalize_fn,
                discard_ambiguous_examples,
            )
            if result is None:
                continue
            return result
        except ValueError:
            if index == len(STRING_NORMALIZATIONS) - 1:
                raise
    return None


def _parse_answer_coordinates(table,
                              answer_texts,
                              discard_ambiguous_examples):
    """Populates answer_coordinates using answer_texts.

    Args:
      table: a Table message, needed to compute the answer coordinates.
      answer_texts: a list of strings
      discard_ambiguous_examples: If true discard if answer has multiple matches.

    Raises:
      ValueError if the conversion fails.
    """

    cost_matrix = _compute_cost_matrix(
        table,
        answer_texts,
        discard_ambiguous_examples,
    )
    if cost_matrix is None:
        return
    row_indices, column_indices = scipy.optimize.linear_sum_assignment(cost_matrix)#solve_dense(cost_matrix)

    # create answer coordinates as list of tuples
    answer_coordinates = []
    for row_index in row_indices:
        flatten_position = column_indices[row_index]
        row_coordinate = flatten_position // len(table.columns)
        column_coordinate = flatten_position % len(table.columns)
        answer_coordinates.append((row_coordinate, column_coordinate))

    return answer_coordinates


### START OF UTILITIES FROM TEXT_UTILS.PY ###

def wtq_normalize(x):
    """Returns the normalized version of x.
    This normalization function is taken from WikiTableQuestions github, hence the
    wtq prefix. For more information, see
    https://github.com/ppasupat/WikiTableQuestions/blob/master/evaluator.py
    Args:
      x: the object (integer type or string) to normalize.
    Returns:
      A normalized string.
    """
    x = x if isinstance(x, six.text_type) else six.text_type(x)
    # Remove diacritics.
    x = "".join(
        c for c in unicodedata.normalize("NFKD", x)
        if unicodedata.category(c) != "Mn")
    # Normalize quotes and dashes.
    x = re.sub(u"[‘’´`]", "'", x)
    x = re.sub(u"[“”]", '"', x)
    x = re.sub(u"[‐‑‒–—−]", "-", x)
    x = re.sub(u"[‐]", "", x)
    while True:
        old_x = x
        # Remove citations.
        x = re.sub(u"((?<!^)\\[[^\\]]*\\]|\\[\\d+\\]|[•♦†‡*#+])*$", "",
                   x.strip())
        # Remove details in parenthesis.
        x = re.sub(u"(?<!^)( \\([^)]*\\))*$", "", x.strip())
        # Remove outermost quotation mark.
        x = re.sub(u'^"([^"]*)"$', r"\1", x.strip())
        if x == old_x:
            break
    # Remove final '.'.
    if x and x[-1] == ".":
        x = x[:-1]
    # Collapse whitespaces and convert to lower case.
    x = re.sub(r"\s+", " ", x, flags=re.U).lower().strip()
    x = re.sub("<[^<]+?>", "", x)
    x = x.replace("\n", " ")
    return x


_TOKENIZER = re.compile(r"\w+|[^\w\s]+", re.UNICODE)


def tokenize_string(x):
    return list(_TOKENIZER.findall(x.lower()))


# List of string normalization functions to be applied in order. We go from
# simplest to more complex normalization procedures.
STRING_NORMALIZATIONS = (
    lambda x: x,
    lambda x: x.lower(),
    tokenize_string,
    wtq_normalize,
)


def to_float32(v):
    """If v is a float reduce precision to that of a 32 bit float."""
    if not isinstance(v, float):
        return v
    return struct.unpack("!f", struct.pack("!f", v))[0]


def convert_to_float(value):
    """Converts value to a float using a series of increasingly complex heuristics.
    Args:
      value: object that needs to be converted. Allowed types include
        float/int/strings.
    Returns:
      A float interpretation of value.
    Raises:
      ValueError if the float conversion of value fails.
    """
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, six.string_types):
        raise ValueError("Argument value is not a string. Can't parse it as float")
    sanitized = value

    try:
        # Example: 1,000.7
        if "." in sanitized and "," in sanitized:
            return float(sanitized.replace(",", ""))
        # 1,000
        #if "," in sanitized and _split_thousands(",", sanitized):
        #    return float(sanitized.replace(",", ""))
        # 5,5556
        #if "," in sanitized and sanitized.count(",") == 1 and not _split_thousands(
        #        ",", sanitized):
        #    return float(sanitized.replace(",", "."))
        # 0.0.0.1
        if sanitized.count(".") > 1:
            return float(sanitized.replace(".", ""))
        # 0,0,0,1
        if sanitized.count(",") > 1:
            return float(sanitized.replace(",", ""))
        return float(sanitized)
    except ValueError:
        # Avoid adding the sanitized value in the error message.
        raise ValueError("Unable to convert value to float")


### END OF UTILITIES FROM TEXT_UTILS.PY ###

def _parse_answer_float(answer_texts, float_value):
    if len(answer_texts) > 1:
        raise ValueError("Cannot convert to multiple answers to single float")
    float_value = convert_to_float(answer_texts[0])
    float_value = float_value

    return answer_texts, float_value


def _has_single_float_answer_equal_to(question, answer_texts, target):
    """Returns true if the question has a single answer whose value equals to target."""
    if len(answer_texts) != 1:
        return False
    try:
        float_value = convert_to_float(answer_texts[0])
        # In general answer_float is derived by applying the same conver_to_float
        # function at interaction creation time, hence here we use exact match to
        # avoid any false positive.
        return to_float32(float_value) == to_float32(target)
    except ValueError:
        return False


def _parse_question(
        table,
        original_question,
        answer_texts,
        answer_coordinates,
        float_value,
        aggregation_function,
        clear_fields,
        discard_ambiguous_examples,
):
    """Parses question's answer_texts fields to possibly populate additional fields.

    Args:
      table: a Pandas dataframe, needed to compute the answer coordinates.
      original_question: a string.
      answer_texts: a list of strings, serving as the answer to the question.
      anser_coordinates:
      float_value: a float, serves as float value signal.
      aggregation_function:
      clear_fields: A list of strings indicating which fields need to be cleared
        and possibly repopulated.
      discard_ambiguous_examples: If true, discard ambiguous examples.

    Returns:
      A Question message with answer_coordinates or float_value field populated.

    Raises:
      ValueError if we cannot parse correctly the question message.
    """
    question = original_question

    # If we have a float value signal we just copy its string representation to
    # the answer text (if multiple answers texts are present OR the answer text
    # cannot be parsed to float OR the float value is different), after clearing
    # this field.
    if "float_value" in clear_fields and float_value is not None:
        if not _has_single_float_answer_equal_to(question, answer_texts, float_value):
            del answer_texts[:]
            float_value = float(float_value)
            if float_value.is_integer():
                number_str = str(int(float_value))
            else:
                number_str = str(float_value)
            answer_texts = []
            answer_texts.append(number_str)

    if not answer_texts:
        raise ValueError("No answer_texts provided")

    for field_name in clear_fields:
        if field_name == "answer_coordinates":
            answer_coordinates = None
        if field_name == "float_value":
            float_value = None
        if field_name == "aggregation_function":
            aggregation_function = None

    error_message = ""
    if not answer_coordinates:
        try:
            answer_coordinates = _parse_answer_coordinates(
                table,
                answer_texts,
                discard_ambiguous_examples,
            )
        except ValueError as exc:
            error_message += "[answer_coordinates: {}]".format(str(exc))
            if discard_ambiguous_examples:
                raise ValueError(f"Cannot parse answer: {error_message}")

    if not float_value:
        try:
            answer_texts, float_value = _parse_answer_float(answer_texts, float_value)
        except ValueError as exc:
            error_message += "[float_value: {}]".format(str(exc))

    # Raises an exception if we cannot set any of the two fields.
    if not answer_coordinates and not float_value:
        raise ValueError("Cannot parse answer: {}".format(error_message))

    return question, answer_texts, answer_coordinates, float_value, aggregation_function


# TODO(piccinno): Use some sort of introspection here to get the field names of
# the proto.
_CLEAR_FIELDS = frozendict.frozendict({
    SupervisionMode.REMOVE_ALL: [
        "answer_coordinates", "float_value", "aggregation_function"
    ],
    SupervisionMode.REMOVE_ALL_STRICT: [
        "answer_coordinates", "float_value", "aggregation_function"
    ]
})


def parse_question(table, question, answer_texts, answer_coordinates=None, float_value=None, aggregation_function=None,
                   mode=SupervisionMode.REMOVE_ALL):
    """Parses answer_text field of a question to populate additional fields required by TAPAS.

    Args:
        table: a Pandas dataframe, needed to compute the answer coordinates. Note that one should apply .astype(str)
        before supplying the table to this function.
        question: a string.
        answer_texts: a list of strings, containing one or more answer texts that serve as answer to the question.
        answer_coordinates: optional answer coordinates supervision signal, if you already have those.
        float_value: optional float supervision signal, if you already have this.
        aggregation_function: optional aggregation function supervised signal, if you already have this.
        mode: see SupervisionMode enum for more information.

    Returns:
        A list with the question, populated answer_coordinates or float_value.

    Raises:
        ValueError if we cannot parse correctly the question string.
    """
    if mode == SupervisionMode.NONE:
        return question, answer_texts

    clear_fields = _CLEAR_FIELDS.get(mode, None)
    if clear_fields is None:
        raise ValueError(f"Mode {mode.name} is not supported")

    return _parse_question(
        table,
        question,
        answer_texts,
        answer_coordinates,
        float_value,
        aggregation_function,
        clear_fields,
        discard_ambiguous_examples=mode == SupervisionMode.REMOVE_ALL_STRICT,
    )



def _get_cell_token_probs(probabilities, segment_ids, row_ids, column_ids):
    for i, p in enumerate(probabilities):
        segment_id = segment_ids[i]
        col = column_ids[i] - 1
        row = row_ids[i] - 1
        if col >= 0 and row >= 0 and segment_id == 1:
            yield i, p

def _get_mean_cell_probs(probabilities, segment_ids, row_ids, column_ids):
    """Computes average probability per cell, aggregating over tokens."""
    coords_to_probs = collections.defaultdict(list)
    for i, prob in _get_cell_token_probs(probabilities, segment_ids, row_ids, column_ids):
        col = column_ids[i] - 1
        row = row_ids[i] - 1
        coords_to_probs[(col, row)].append(prob)
    return {coords: np.array(cell_probs).mean() for coords, cell_probs in coords_to_probs.items()}


def convert_logits_to_predictions(data, logits, logits_agg=None, cell_classification_threshold=0.5):
    """
    Converts logits of :class:`~transformers.TapasForQuestionAnswering` to actual predicted answer coordinates and
    optional aggregation indices.

    The original implementation, on which this function is based, can be found `here
    <https://github.com/google-research/tapas/blob/4908213eb4df7aa988573350278b44c4dbe3f71b/tapas/experiments/prediction_utils.py#L288>`__.

    Args:
        data (:obj:`dict`):
            Dictionary mapping features to actual values. Should be created using
            :class:`~transformers.TapasTokenizer`.
        logits (:obj:`np.ndarray` of shape ``(batch_size, sequence_length)``):
            Tensor containing the logits at the token level.
        logits_agg (:obj:`np.ndarray` of shape ``(batch_size, num_aggregation_labels)``, `optional`):
            Tensor containing the aggregation logits.
        cell_classification_threshold (:obj:`float`, `optional`, defaults to 0.5):
            Threshold to be used for cell selection. All table cells for which their probability is larger than
            this threshold will be selected.

    Returns:
        :obj:`tuple` comprising various elements depending on the inputs:

        - predicted_answer_coordinates (``List[List[[tuple]]`` of length ``batch_size``): Predicted answer
          coordinates as a list of lists of tuples. Each element in the list contains the predicted answer
          coordinates of a single example in the batch, as a list of tuples. Each tuple is a cell, i.e. (row index,
          column index).
        - predicted_aggregation_indices (``List[int]``of length ``batch_size``, `optional`, returned when
          ``logits_aggregation`` is provided): Predicted aggregation operator indices of the aggregation head.
    """
    # input data is of type float32
    # np.log(np.finfo(np.float32).max) = 88.72284
    # Any value over 88.72284 will overflow when passed through the exponential, sending a warning
    # We disable this warning by truncating the logits.
    logits[logits < -88.7] = -88.7

    # Compute probabilities from token logits
    probabilities = 1 / (1 + np.exp(-logits)) * data["attention_mask"]
    token_types = [
        "segment_ids",
        "column_ids",
        "row_ids",
        "prev_labels",
        "column_ranks",
        "inv_column_ranks",
        "numeric_relations",
    ]

    # collect input_ids, segment ids, row ids and column ids of batch. Shape (batch_size, seq_len)
    input_ids = data["input_ids"]
    segment_ids = data["token_type_ids"][:, :, token_types.index("segment_ids")]
    row_ids = data["token_type_ids"][:, :, token_types.index("row_ids")]
    column_ids = data["token_type_ids"][:, :, token_types.index("column_ids")]

    # next, get answer coordinates for every example in the batch
    num_batch = input_ids.shape[0]
    predicted_answer_coordinates = []
    all_cell_coords_to_prob = []
    for i in range(num_batch):
        probabilities_example = probabilities[i].tolist()
        segment_ids_example = segment_ids[i]
        row_ids_example = row_ids[i]
        column_ids_example = column_ids[i]

        max_width = column_ids_example.max()
        max_height = row_ids_example.max()

        if max_width == 0 and max_height == 0:
            continue

        cell_coords_to_prob = _get_mean_cell_probs(
            probabilities_example,
            segment_ids_example.tolist(),
            row_ids_example.tolist(),
            column_ids_example.tolist(),
        )
        all_cell_coords_to_prob.append(cell_coords_to_prob)

        # Select the answers above the classification threshold.
        answer_coordinates = []
        for col in range(max_width):
            for row in range(max_height):
                cell_prob = cell_coords_to_prob.get((col, row), None)
                if cell_prob is not None:
                    if cell_prob > cell_classification_threshold:
                        answer_coordinates.append((row, col))
        answer_coordinates = sorted(answer_coordinates)
        predicted_answer_coordinates.append(answer_coordinates)

    output = (predicted_answer_coordinates,)

    if logits_agg is not None:
        predicted_aggregation_indices = logits_agg.argmax(dim=-1)
        output = (predicted_answer_coordinates, predicted_aggregation_indices.tolist())

    return output, all_cell_coords_to_prob


def find_answer_cells(out, tables, cells_probs):
    operations = {0: "SELECT", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
    answers = []
    all_cells = []
    all_op = []

    for i in range(len(out[1])):
        lst = out[0][i]
        op_index = out[1][i]
        cells = []
        cells_p = []
        for row_index, column_index in lst:
            val = tables[i].iloc[row_index, column_index]
            cells.append(val)
            cells_p.append(cells_probs[i][(column_index, row_index)])
        op = operations[op_index]
        try:
            if op == 'SELECT':
                # if len(cells) > 1:
                #     float_cells = [float(x) for x in cells]
                #     sorted_float_cells = [x for _, x in sorted(zip(cells_p, float_cells))]
                #     top_two_cells = sorted_float_cells[-2:]
                #     final_answer = max(top_two_cells) - min(top_two_cells)
                # else:
                final_answer = cells[0]

            elif op == 'SUM':
                final_answer = sum([float(x) for x in cells])
            elif op == 'AVERAGE':
                final_answer = sum([float(x) for x in cells]) / len(cells)
            elif op == 'COUNT':
                final_answer = len(cells)
        except:
            final_answer = None
        answers.append(final_answer)
        all_cells.append(cells)
        all_op.append(op)
    return answers, all_cells, all_op
