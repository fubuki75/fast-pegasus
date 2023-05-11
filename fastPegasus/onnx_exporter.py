from .huggingface_utils import get_auth_token
from .onnx_models_structure import (
    PegasusEncoder,
    PegasusDecoderWithLMhead,
    PegasusDecoderWithLMheadInitial,
)
from transformers import (
    AutoConfig,
    PegasusForConditionalGeneration,
)
import torch
import functools
import operator
from progress.bar import Bar
from pathlib import Path
import os

_folder = Path.cwd()
saved_models_path = _folder.joinpath("models")

Bar.check_tty = False


def create_pegasus_encoder_decoder(pretrained_version="google/pegasus-cnn_dailymail"):
    """Generates an encoder and a decoder model with a language model head from a pretrained huggingface model

    Args:
        pretrained_version (str): Name of a pretrained model, or path to a pretrained / finetuned version of Pegasus

    Returns:
        simplified_encoder: pytorch Pegasus encoder with a wrapper to output only the hidden states
        decoder_with_lm_head: pytorch Pegasus decoder with a language modeling head
    """

    model = PegasusForConditionalGeneration.from_pretrained(pretrained_version, use_auth_token=get_auth_token())

    return turn_model_into_encoder_decoder(model)


def turn_model_into_encoder_decoder(model):

    encoder = model.get_encoder()
    decoder = model.get_decoder()
    lm_head = model.get_output_embeddings()
    final_logits_bias = model.final_logits_bias # lm_head projection bias, check the forard func of PegasusForConditionalGeneration in `modeling_pegasus.py`

    decoder_with_lm_head = PegasusDecoderWithLMhead(decoder, lm_head, final_logits_bias, model.config)
    simplified_encoder = PegasusEncoder(encoder)
    decoder_with_lm_head_init = PegasusDecoderWithLMheadInitial(decoder, lm_head, final_logits_bias, model.config)

    return simplified_encoder, decoder_with_lm_head, decoder_with_lm_head_init


def generate_onnx_representation(
    pretrained_version=None,
    model=None,
    output_path=None,
    input_sequence_length=256,
    onnx_opset_version=12,  # no other opset versions are tested, change at your own risk
):
    """Exports a given huggingface pretrained model, or a given model and tokenizer, to onnx

    Args:
        pretrained_version (str): Name of a pretrained model, or path to a pretrained / finetuned version of Pegasus
        output_path (Optional[str]): if missing then use ./models
        input_sequence_length (Optional[int]): typical input sequence length, for use by the ORT for possible optimization
        onnx_opset_version (Optional[int]): ONNX Operator Set Version, default 12 is the only tested version
    """
    if (pretrained_version is None) and model is None:
        print(
            "You need to specify pretrained_version of Pegasus (the pretrained model you wish to export). Alternatively you can export a model you have in memory."
        )
        return

    if model is not None:
        (
            simplified_encoder,
            decoder_with_lm_head,
            decoder_with_lm_head_init,
        ) = turn_model_into_encoder_decoder(model)
    else:
        (
            simplified_encoder,
            decoder_with_lm_head,
            decoder_with_lm_head_init,
        ) = create_pegasus_encoder_decoder(pretrained_version)

    # model paths for enc, dec and dec_init
    output_path = saved_models_path if output_path is None else Path(output_path)
    encoder_path, decoder_path, init_decoder_path = get_model_paths(
        pretrained_version, output_path, quantized=False
    )

    model_config = AutoConfig.from_pretrained(pretrained_version, use_auth_token=get_auth_token())

    # Though these are dummy inputs, ORT optimizations do reference these values,
    # so it is worth using values as close to production as possible

    # a dummy input seq for encoder
    batch_size = 1  # not configurable since only CPU
    enc_seq_length = input_sequence_length
    # a dummy input for decoder input，this length is always one because it's just the last generated token
    dec_seq_length = 1

    # encoder mask and input_id
    input_ids = torch.ones(batch_size, enc_seq_length, dtype=torch.int64)
    attention_mask = torch.ones(batch_size, enc_seq_length, dtype=torch.int64)

    n_heads = model_config.decoder_attention_heads
    d_kv = model_config.d_model // n_heads

    #decoder mask and input_id
    input_ids_dec = torch.ones(batch_size, dec_seq_length, dtype=torch.int64)  # (batch,1)
    attention_mask_dec = torch.ones(batch_size, enc_seq_length, dtype=torch.int64) # decoder mask will be transformed from (batch,enc_len) to (batch,1,1(dec_len),enc_len)

    # enc_out for decoder input
    enc_out = torch.ones(
        (batch_size, enc_seq_length, model_config.d_model), dtype=torch.float32
    ) # (batch,src_seq_len,d_model)

    # self_attention_past_key_values = torch.ones(
    #     (model_config.num_decoder_layers, 2, batch_size, n_heads, seq_length_a, d_kv), dtype=torch.float32)
    # cross_attention_past_key_values = torch.ones(
    #     (model_config.num_decoder_layers, 2, batch_size, n_heads, seq_length_b, d_kv), dtype=torch.float32)

    '''
    prepare for past_key_value, why this param is needed is shown in the following link: this design is quite common in transformers for speed up decoding process
    https://huggingface.co/docs/transformers/main/en/model_doc/t5#transformers.T5ForConditionalGeneration.forward.past_key_values
    shape: num_decoder_layer tuples. each consists of 4 tensors of shape (batch_size, num_heads, sequence_length - 1(waiting for the latest token for concatenation), embed_size_per_head)
    (batch_size, num_heads, sequence_length - 1, embed_size_per_head) is the shape of keys and values after projection
    '''
    sa = torch.ones(
        (batch_size, n_heads, dec_seq_length, d_kv), dtype=torch.float32
    )  # 1, 8, 1, 64
    ca = torch.ones(
        (batch_size, n_heads, enc_seq_length, d_kv), dtype=torch.float32
    )  # 1, 8, variable, 64
    pegasus_block = (sa, sa, ca, ca)
    past_key_values = (pegasus_block,) * model_config.decoder_layers # past_key_values for each layer
    # flatten
    flat_past_key_values = functools.reduce(operator.iconcat, past_key_values, []) # (24,...)

    decoder_all_inputs = tuple(
        [input_ids_dec, attention_mask_dec, enc_out] + flat_past_key_values
    ) # all done for decoder input

    # for progress bars
    bar = Bar("Exporting to onnx...", max=3)

    import warnings

    # ignores all the warnings during conversion
    warnings.filterwarnings("ignore")

    # Exports to ONNX
    with torch.no_grad():

        # the param of input_names(list) in onnx.export func
        decoder_inputs = [
            "input_ids",
            "encoder_attention_mask",
            "encoder_hidden_states",
        ]
        pkv_input_names = ["pkv_{}".format(i) for i in range(len(flat_past_key_values))]
        decoder_input_names = decoder_inputs + pkv_input_names

        # the param of output_names(list) in onnx.export
        decoder_output_names = ["logits", "output_past_key_values"]

        # the param of dynamic_axes to point out the dynamic dimension, dict of dict {'param name':{'dim_id':'dim name'}}
        dyn_axis_general = {0: "batch", 1: "sequence"}
        dyn_axis_pkv = {0: "batch", 2: "seq_length"}
        dyn_axis = {
            "input_ids": dyn_axis_general,
            "encoder_attention_mask": dyn_axis_general,
            "encoder_hidden_states": dyn_axis_general,
            "logits": dyn_axis_general,
            "output_past_key_values": dyn_axis_general,
        }
        dyn_pkv = {
            "pkv_{}".format(i): dyn_axis_pkv
            for i in range(len(flat_past_key_values))
        }
        dyn_axis_params = {**dyn_axis, **dyn_pkv}

        # decoder to utilize past key values:
        torch.onnx.export(
            decoder_with_lm_head,
            decoder_all_inputs, # tuple multiple input elements
            decoder_path.as_posix(),
            export_params=True,
            do_constant_folding=True,
            opset_version=onnx_opset_version,
            input_names=decoder_input_names,
            output_names=decoder_output_names,
            dynamic_axes=dyn_axis_params,
        )
        bar.next()

        torch.onnx.export(
            simplified_encoder, # encoder
            args=(input_ids, attention_mask),
            f=encoder_path.as_posix(),
            export_params=True,
            opset_version=onnx_opset_version,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["hidden_states"],
            dynamic_axes={
                "input_ids": dyn_axis_general,
                "attention_mask": dyn_axis_general,
                "hidden_states": dyn_axis_general,
            },
        )
        bar.next()
        # initial decoder to produce past key values
        torch.onnx.export(
            decoder_with_lm_head_init,
            (input_ids_dec, attention_mask_dec, enc_out), # no past_key_value
            init_decoder_path.as_posix(),
            export_params=True,
            opset_version=onnx_opset_version,
            input_names=[
                "input_ids",
                "encoder_attention_mask",
                "encoder_hidden_states",
            ],
            output_names=["logits", "past_key_values"],
            dynamic_axes={
                # batch_size, seq_length = input_shape
                "input_ids": dyn_axis_general,
                "encoder_attention_mask": dyn_axis_general,
                "encoder_hidden_states": dyn_axis_general,
                "logits": dyn_axis_general,
                "past_key_values": dyn_axis_general,
            },
        )
        bar.next()
        bar.finish()

    return encoder_path, decoder_path, init_decoder_path


def get_model_paths(pretrained_model, model_path, quantized):

    model_path.mkdir(parents=True, exist_ok=True)

    # gets only the filename
    pretrained_model_name = Path(pretrained_model).stem

    if not quantized:
        encoder_path = model_path.joinpath(f"{pretrained_model_name}-encoder.onnx")
        decoder_path = model_path.joinpath(f"{pretrained_model_name}-decoder.onnx")
        init_decoder_path = model_path.joinpath(
            f"{pretrained_model_name}-init-decoder.onnx"
        )
    else:
        encoder_path = model_path.joinpath(
            f"{pretrained_model_name}-encoder-quantized.onnx"
        )
        decoder_path = model_path.joinpath(
            f"{pretrained_model_name}-decoder-quantized.onnx"
        )
        init_decoder_path = model_path.joinpath(
            f"{pretrained_model_name}-init-decoder-quantized.onnx"
        )

    return encoder_path, decoder_path, init_decoder_path


def quantize(models_name_or_path):
    """
    float32-->int8
    Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU

    Uses unsigned ints for activation values, signed ints for weights, per
    https://onnxruntime.ai/docs/performance/quantization.html#data-type-selection
    it is faster on most CPU architectures
    Args:
        onnx_model_path: Path to location the exported ONNX model is stored
    Returns: The Path generated for the quantized
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    bar = Bar("Quantizing...", max=3)

    quant_model_paths = []
    for model in models_name_or_path:
        model_name = model.as_posix()
        output_model_name = f"{model_name[:-5]}-quantized.onnx"
        quantize_dynamic(
            model_input=model_name,
            model_output=output_model_name,
            per_channel=True,
            reduce_range=True, # should be the same as per_channel
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,  # per docs, signed is faster on most CPUs
            optimize_model=False,
        )  # op_types_to_quantize=['MatMul', 'Relu', 'Add', 'Mul' ],
        quant_model_paths.append(output_model_name)
        bar.next()

    bar.finish()

    return tuple(quant_model_paths)
