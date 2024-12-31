import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from base_bert import BertPreTrainedModel
from utils import *
import inspect
from inspect import currentframe, getframeinfo


class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    
    self.position_buckets = 256
    self.bucket_size = 256
    self.max_position_embeddings = 512
    self.max_relative_position = 512
    self.pos_ebd_size = 512
    self.num_attention_heads = config.num_attention_heads # 12
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # 768/12=64
    self.all_head_size = self.num_attention_heads * self.attention_head_size # 12*64=768
    self.pos_embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.pos_dropout = nn.Dropout(config.hidden_dropout_prob)

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=True) # hidden_size = 768, all_head_size = 768
    self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
    self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
    self.pos_query = nn.Linear(config.hidden_size, self.all_head_size)
    self.pos_key = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    self.relative_position_embeddings = nn.Embedding(2*self.max_relative_position, self.all_head_size)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    bs, seq_len = x.shape[:2]
    #print(get_line_number(),, x.shape, "bs, seq_len, hidden_state")
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = proj.transpose(1, 2)
    #print(get_line_number(),, proj.shape, "bs, num_attention_heads, seq_len, attention_head_size")
    return proj
  def transpose_for_scores(self, x, attention_heads):
    new_x_shape = x.size()[:-1] + (attention_heads, -1)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
  
  def attention(self, content_key_layer, content_query_layer, value_layer, attention_mask, relative_pos, rel_embeddings):
    # Each attention is calculated following eq. (1) of https://arxiv.org/pdf/1706.03762.pdf.
    # Attention scores are calculated by multiplying the key and query to obtain
    # a score matrix S of size [bs, num_attention_heads, seq_len, seq_len].
    # S[*, i, j, k] represents the (unnormalized) attention score between the j-th and k-th
    # token, given by i-th attention head.
    # Before normalizing the scores, use the attention mask to mask out the padding token scores.
    # Note that the attention mask distinguishes between non-padding tokens (with a value of 0)
    # and padding tokens (with a value of a large negative number).

    # Make sure to:
    # - Normalize the scores with softmax.
    # - Multiply the attention scores with the value to get back weighted values.
    # - Before returning, concatenate multi-heads to recover the original shape:
    #   [bs, seq_len, num_attention_heads * attention_head_size = hidden_size].

    ### TODO
    softmax = nn.Softmax(dim=3)
    if rel_embeddings is None:
      rel_embeddings = self.rel_embeddings.weight
      # print(get_line_number(), "rel_embeddings shape is:", rel_embeddings.shape)
    # Last two dimension of key needs to match second last dimension of query
    bs, num_attention_heads, seq_len, attention_head_size = content_key_layer.shape
    S_content = content_query_layer@content_key_layer.transpose(-1,-2)
    # print(get_line_number(), "S_content shape is:", S_content.shape, "attention_mask shape is:", attention_mask.shape, "content_query_layer shape is:", content_query_layer.shape, "content_key_layer shape is:", content_key_layer.shape)
    S_content = S_content.div(math.sqrt((3*self.num_attention_heads)))
    # compute positional attention scores
    rel_embeddings = rel_embeddings.to(self.device)
    rel_embeddings = self.pos_dropout(rel_embeddings)
    S_position = self.PositionToContentAttention(content_key_layer, content_query_layer, relative_pos, rel_embeddings)
    score = S_content + S_position
    score = score + attention_mask
    score_normalized = softmax(score)
    score_normalized_dropout = self.dropout(score_normalized)
    attn_value = score_normalized_dropout@value_layer
    attn_value = attn_value.transpose(1,2).contiguous()
    # Concatenate multi-heads
    attn_value = attn_value.view(bs,seq_len, num_attention_heads*attention_head_size)
    return attn_value
  def PositionToContentAttention(self, content_key_layer, content_query_layer, relative_pos, rel_embeddings):
    bs, num_attn_heads, seq_len, attention_head_size = content_query_layer.shape
    if relative_pos is None:
      relative_pos = build_relative_position(seq_len, seq_len, bucket_size=self.bucket_size, max_position=self.max_position_embeddings, device=self.device)
      if relative_pos.dim()==2:
        relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
      elif relative_pos.dim()==3:
        relative_pos = relative_pos.unsqueeze(1)
    att_span = self.max_position_embeddings
    rel_embeddings = rel_embeddings[self.max_position_embeddings-att_span:self.max_position_embeddings+att_span, :].unsqueeze(0)
    #print(get_line_number(),, relative_pos.shape, "content_querpy_layer shape is:", content_query_layer.shape, "content_key_layer shape is:", content_key_layer.shape)
    pos_key_layer = self.transform(rel_embeddings, self.pos_key).squeeze(0).repeat(content_query_layer.size(0), 1, 1).view(bs, num_attn_heads, 2*self.max_position_embeddings, self.attention_head_size)
    pos_query_layer = self.transform(rel_embeddings, self.pos_key).squeeze(0).repeat(content_query_layer.size(0), 1, 1).view(bs, num_attn_heads, 2*self.max_position_embeddings, self.attention_head_size)
    #print(get_line_number(),, rel_pos_embed.shape)
    # content to position 
    #print(get_line_number(),, content_query_layer.shape, "position_key_layer shape is:", position_key_layer.shape)
    c2p_att = content_query_layer@(pos_key_layer.transpose(-1, -2)/math.sqrt(self.all_head_size*3))
    c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span*2-1).squeeze(0).expand([bs, self.num_attention_heads, seq_len, seq_len])
    c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos)
    #print(get_line_number(),, c2p_attention_score.shape)
    # position to content
    p2c_att = (pos_query_layer/math.sqrt(3*self.num_attention_heads))@content_key_layer.transpose(-1, -2)
    p2c_att = torch.gather(p2c_att, dim=-2, index=c2p_pos)
    
    score = c2p_att + p2c_att

    return score
  
  def forward(self, hidden_states, attention_mask, relative_pos=None):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    #print(get_line_number(),, "hidden_states shape is:", hidden_states.shape, "[bs, seq_len, hidden_state]")
    content_query_layer = self.transform(hidden_states, self.query).float()
    content_key_layer = self.transform(hidden_states, self.key).float()
    value_layer = self.transform(hidden_states, self.value)
    rel_embeddings = self.relative_position_embeddings.weight
    # Calculate the multi-head attention.
    attn_value = self.attention(content_key_layer, content_query_layer, value_layer, attention_mask, relative_pos, rel_embeddings)
    return attn_value
  
class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = BertSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    This function is applied after the multi-head attention layer or the feed forward layer.
    input: the input of the previous layer
    output: the output of the previous layer
    dense_layer: used to transform the output
    dropout: the dropout to be applied 
    ln_layer: the layer norm to be applied
    """
    # Hint: Remember that BERT applies dropout to the transformed output of each sub-layer,
    # before it is added to the sub-layer input and normalized with a layer norm.
    ### TODO
    dense = dense_layer(output)
    dropout_dense = dropout(dense)
    output = ln_layer(input + dropout_dense)
    return output


  def forward(self, hidden_states, attention_mask, rel_embeddings):
    """
    hidden_states: either from the embedding layer (first BERT layer) or from the previous BERT layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf.
    Each block consists of:
    1. A multi-head attention layer (BertSelfAttention).
    2. An add-norm operation that takes the input and output of the multi-head attention layer.
    3. A feed forward layer.
    4. An add-norm operation that takes the input and output of the feed forward layer.
    """
    ### TODO
    # Multi-head attention
    attn_layer = self.self_attention(hidden_states,attention_mask, rel_embeddings)

    # First add & norm
    add_norm_layer_1 = self.add_norm(hidden_states, attn_layer, self.attention_dense, self.attention_dropout, self.attention_layer_norm)

    # Feed forward 
    feed_forward_layer = self.interm_dense(add_norm_layer_1)
    feed_forward_layer = self.interm_af(feed_forward_layer)

    # Second add & norm
    out = self.add_norm(add_norm_layer_1, feed_forward_layer, self.out_dense, self.out_dropout, self.out_layer_norm)

    return out


class BertModel(BertPreTrainedModel):
  """
  The BERT model returns the final embeddings for each token in a sentence.
  
  The model consists of:
  1. Embedding layers (used in self.embed).
  2. A stack of n BERT layers (used in self.encode).
  3. A linear transformation layer for the [CLS] token (used in self.forward, as given).
  """
  def __init__(self, config):
    super().__init__(config)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    self.config = config
    self.position_buckets = 256
    self.bucket_size = 256
    self.max_position_embeddings = config.max_position_embeddings
    self.max_relative_position = 512

    # Embedding layers.
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size) 
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Register position_ids (1, len position emb) to buffer because it is a constant.
    position_ids = torch.arange(self.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # BERT encoder.
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # [CLS] token transformations.
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    batch_size, seq_length = input_shape[:2]

    # Get word embedding from self.word_embedding into input_embeds.
    ### TODO
    #print(get_line_number(),, "input_shape is:", input_shape, "[batch_size, seq_len]")
    inputs_embeds = self.word_embedding(input_ids)
    #print(get_line_number(),, "after transformation, inputs_embeds's shape is:", inputs_embeds.shape, "[batch_size, seq_len, hidden_size]")

    # Get token type ids. Since we are not considering token type, this embedding is
    # just a placeholder.
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)
    # Use pos_ids to get position embedding from self.pos_embedding into pos_embeds.
    pos_ids = self.position_ids[:, :seq_length]
    ### TODO
    pos_embeds = self.pos_embedding(pos_ids)
    # Add three embeddings together; then apply ebreakpoint()mbed_layer_norm and dropout and return.
    ### TODO
    added_embeddings = inputs_embeds + tk_type_embeds + pos_embeds

    # Apply layer norm and dropout
    embeddings_normalized = self.embed_layer_norm(added_embeddings)
    embeddings_normalized_dropout = self.embed_dropout(embeddings_normalized)

    return embeddings_normalized_dropout


  def encode(self, hidden_states, attention_mask, input_ids):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # Get the extended attention mask for self-attention.
    # Returns extended_attention_mask of size [batch_size, 1, 1, seq_len].
    # Distinguishes between non-padding tokens (with a value of 0) and padding tokens
    # (with a value of a large negative number).
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)
    # Pass the hidden states through the encoder layers.
    for i, layer_module in enumerate(self.bert_layers):
      # Feed the encoding from the last bert_layer to the next.
      hidden_states = layer_module(hidden_states, extended_attention_mask, rel_embeddings=None)
    return hidden_states 

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # Get the embedding for each input token.
    embedding_output = self.embed(input_ids=input_ids)

    # Feed to a transformer (a stack of BertLayers).
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask, input_ids=input_ids)

    # Get cls token hidden state.
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}

