import torch

class PegasusDecoderWithLMhead(torch.nn.Module):
    """ Creation of a class to combine the decoder and the lm head """

    def __init__(self, decoder, lm_head, final_logits_bias, config):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.final_logits_bias = final_logits_bias
        self.config = config

    def forward(self, *inputs):
        # inputs[:3] the first three elements are: input_ids_dec, attention_mask_dec, enc_out, check `onnx_exporter.py` decoder_all_inputs
        input_ids, attention_mask, encoder_hidden_states = inputs[:3]

        list_pkv = inputs[3:] # flat_past_key_values, 6*4 tensors with the shape of (batch,head,seq_len,dv)
        # transform flat_past_key_values to past_key_values
        past_key_values = tuple(list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4))

        # modeling_pegasus.py ln 1041 which calls the `forward` function of class `PegasusDecoder`
        decoder_output = self.decoder(
            input_ids=input_ids,  # decoder_input_ids
            encoder_attention_mask=attention_mask, # encoder_atn_mask (sounds wierd?) is attention mask of decoder, since this is the only param concerning attention mask for PegasusDecoder forward func
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
        )

        # decoder_output consists of two elements: [0]last_hidden_state,[1]past_key_values
        lm_head_out = self.lm_head(decoder_output[0]) + self.final_logits_bias
        # return lm_output and the updated past_key_values
        return lm_head_out, decoder_output[1]

class PegasusEncoder(torch.nn.Module):
    """ Creation of a class to output only the last hidden state from the encoder """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, *input, **kwargs): # input_ids, attention_mask
        return self.encoder(*input, **kwargs)[0] # (batch,seq_len,d_model)


class PegasusDecoderWithLMheadInitial(torch.nn.Module):
    """ Creation of a class to combine the decoder and the lm head """

    def __init__(self, decoder, lm_head, final_logits_bias, config):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.final_logits_bias = final_logits_bias
        self.config = config

    def forward(self, input_ids, attention_mask, encoder_hidden_states):
        decoder_output = self.decoder(
            input_ids=input_ids,
            encoder_attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
        )

        return (
            self.lm_head(decoder_output[0]) + self.final_logits_bias,
            decoder_output[1], # return the first past_key_values
        )
