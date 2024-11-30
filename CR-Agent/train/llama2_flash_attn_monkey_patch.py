"""
This file is copied from: https://github.com/lm-sys/FastChat
"""
import warnings
from typing import Optional, Tuple
#Optional：表示一个值可以是指定的类型，也可以是None。它用于表示一个变量可能没有值的情况。例如，Optional[int]表示一个变量可以是int类型，也可以是None。
#Tuple：表示一个元组（tuple），它是一个不可变的序列，可以包含任意类型的元素。Tuple可以有固定数量的元素，也可以有可变数量的元素。例如，Tuple[int, str]表示一个包含一个整数和一个字符串的元组。
#通过导入这两个类型提示，你可以在代码中使用它们来为变量、函数参数和返回值添加类型注解，从而提高代码的可读性和健壮性。

import torch
from flash_attn.bert_padding import pad_input, unpad_input
#与BERT模型相关的数据填充和解填充操作。
#根据提供的搜索结果，flash_attn在不同的GPU上相比于PyTorch标准注意力可以实现显著的速度提升和内存节省。它特别适用于处理长序列，因为其内存占用是线性的，而标准注意力的内存占用是二次方的。
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,#用于计算变长序列，通过将键和值打包成一个张量作为输入，避免了显式连接K、V的梯度，从而提高了计算效率。
)
from transformers.models.llama.modeling_llama import (#modeling_llama则是包含该模型实现的模块
    LlamaAttention,
    LlamaModel,
    rotate_half,#用于对输入的张量进行旋转操作，这在某些类型的注意力机制中用于增强模型对序列中不同位置信息的处理能力。
)


def apply_rotary_pos_emb(q, k, cos_sin, position_ids):
    gather_indices = position_ids[:, :, None, None]  # [bsz, seq_len, 1, 1]这行代码创建了一个形状为[bsz, seq_len, 1, 1]的gather_indices张量，其中bsz是批量大小，seq_len是序列长度。这个张量用于后续的索引操作。
    #[:, :, None, None] 是一个索引操作，它告诉PyTorch在 position_ids 张量的基础上增加两个新的维度。 position_ids原本的向量是[bsz, seq_len]
    #: 表示选择所有的行和列，即保持原有的 bsz 和 seq_len 维度不变。
    #None 在Python中是一个特殊的值，当用在张量索引中时，它会被解释为增加一个新的维度。这里使用了两次 None，意味着在原有的两个维度后面各增加了一个新维度。
    #gather_indices 通常指的是一个索引张量，它用于从另一个张量中“收集”特定的元素。
    
    gather_indices = gather_indices.repeat(
        1, 1, cos_sin[0].shape[1], cos_sin[0].shape[3]
    )
    #1, 1,表示gather_indices前两个维度不变换
    #cos_sin[0].shape[1]：第三个维度上重复的次数是cos_sin[0]张量第二维的大小。
    #cos_sin是一个包含余弦和正弦值的张量列表，这里取第一个元素（即余弦值）的第二维大小作为重复次数。
    #cos_sin[0].shape[3]：第四个维度上重复的次数是cos_sin[0]张量第四维的大小。
    
    bsz = gather_indices.shape[0] #gather_indices.shape是一个元组，表示gather_indices的各个维度的大小。
                                  #gather_indices.shape[0]表示gather_indices的第一个维度的大小。
    cos, sin = (
        torch.gather(x.transpose(1, 2).repeat(bsz, 1, 1, 1), 1, gather_indices)
        #将张量 x 的第1维和第2维交换。然后将交换后的张量在第1维上重复 bsz 次。然后把第二维（第二列位置是1）根据gather_indices的索引值在x中选择
        #x是从哪里获得的此时未知
        #repeat的时候第一维是数的重复，第二维是一维列表的重复
        for x in cos_sin
        #列表推导式
    )
    q, k = ((x * cos) + (rotate_half(x) * sin) for x in (q, k))
    #对于变量q和k中的每个对应元素，计算一个基于余弦和正弦函数的数学表达式，然后将计算结果分别赋值回q和k。这是一种同时更新两个变量的简洁写法。
    return q, k


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,#= None表示attention_mask参数的默认值是None。如果调用函数时没有提供这个参数，它将自动被设置为None
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:#-> 表示返回值
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )#这个函数的作用是发出一个警告信息，告诉用户或者开发者一些重要的信息，但不会中断程序的执行。
        #在这个特定的情况下，警告信息是：“Output attentions is not supported for patched LlamaAttention, returning None instead.” 这句话的意思是，
        #对于被修补（patched）的LlamaAttention模型，不支持输出注意力（attentions）权重，因此会返回None。

    bsz, q_len, _ = hidden_states.size() #q_len：代表序列长度（query length），即每个数据样本中的元素数量。
                                        #_：这是一个占位符变量，用于接收剩余的维度大小，但不会在后续代码中使用。
    kv_heads = getattr(self, "num_key_value_heads", self.num_heads)#尝试从self对象中获取名为"num_key_value_heads"的属性值，如果该属性存在，则将其值赋给变量kv_heads；
                                #如果不存在，则将self.num_heads的值赋给kv_heads。这是一种常见的在Python中处理对象属性的写法，允许代码在属性不存在时有一个默认的回退值。

    q, k, v = (
        op(hidden_states).view(bsz, q_len, nh, self.head_dim)
        #这里的op是一个函数，它将hidden_states（隐藏状态）作为输入，然后通过.view方法将输出重塑为一个新的形状。
        #bsz是批量大小（batch size），q_len是序列长度，nh是头的数量（对于q、k和v，这里分别对应self.num_heads、kv_heads和kv_heads），self.head_dim是每个头的维度。
        
        for op, nh in (
            (self.q_proj, self.num_heads),#self.q_proj是一个函数，用于将输入的隐藏状态投影到查询（q）的空间，self.num_heads是查询头的数量。
            (self.k_proj, kv_heads),
            (self.v_proj, kv_heads),
        )
        #循环中的op代表这三个函数中的一个，而nh代表对应的头的数量。
        #这个循环将对每个函数和头的数量进行迭代，对隐藏状态进行相应的投影操作，以生成查询、键和值的表示，这些表示将被用于后续的自注意力计算。
    )
    # shape: (b, s, num_heads, head_dim)

    kv_seq_len = k.shape[1]#第一维是bsz，第二维是q len
    past_kv_len = 0
    if past_key_value is not None:
        past_kv_len = past_key_value[0].shape[1]
        kv_seq_len += past_kv_len
    #past_key_value 通常用于存储之前计算的键（key）和值（value），这样可以在处理序列数据时复用这些值，特别是在解码器或者循环处理序列数据时。
    
    cos_sin = self.rotary_emb(v, seq_len=kv_seq_len)#这行代码调用了一个名为 rotary_emb 的方法，这个方法可能是一个自定义的函数，用于给值（value）向量 v 应用
    #旋转位置编码（rotary position embedding）。这种编码是一种相对位置编码的变体，它可以帮助模型捕捉序列中不同位置之间的关系。
    q, k = apply_rotary_pos_emb(q, k, cos_sin, position_ids)

    if past_key_value is not None:
        # reuse k, v
        k = torch.cat([past_key_value[0], k], dim=1)
        #这段代码的作用是将 past_key_value 中的第一个张量和另一个张量 k 沿着第二个维度连接起来，并将结果赋值给变量 k
        v = torch.cat([past_key_value[1], v], dim=1)

    past_key_value = (k, v) if use_cache else None

    key_padding_mask = attention_mask
    # Ideally we could just do this:
    #  q, indices, cu_q_lens, max_s = unpad_input(q, key_padding_mask[:, -q_len:])unpad_input函数的作用是去除填充，返回未填充的序列、索引、累积长度和最大长度。
    # but this does not work as Flash attention treats the q seq and kv seq as starting at index 0
    # which then breaks the causality logic. Probably if q_len >> past_kv_len we should
    # just skip flash attention. Leaving this in for now to demonstrate correctness of
    # flash attention information even when q needs padding.
    #稀疏注意力模式：Flash Attention采用稀疏的注意力模式，这意味着它只在序列中的特定位置之间计算注意力，而不是在所有位置之间都计算。这减少了计算量和内存需求。
    #快速近似：Flash Attention使用快速近似方法来计算注意力权重，这可以减少计算复杂度。
    # TODO(siddartha): delegate back to original implementation on this condition.
    if past_kv_len > 0:
        q = torch.cat(
            (
                torch.full(
                    (bsz, past_kv_len, self.num_heads, self.head_dim),
                    0.0,
                    dtype=q.dtype,
                    device=q.device,
                ),
                q,
            ),
            dim=1,
        )
        #这段代码的作用是，如果存在过去的键值对，就在当前查询向量q的前面添加一个全0的序列，以便在自注意力计算中考虑到过去的信息。
        #这样做可以帮助模型在处理序列数据时，如解码器或循环神经网络中，复用之前计算的键值对

    if key_padding_mask is None:#键填充掩码通常用于指示序列中哪些位置是填充的（即不包含有效信息），哪些是实际的数据。如果key_padding_mask为None，表示没有填充，或者不需要考虑填充。
        output = flash_attn_func(q, k, v, 0.0, softmax_scale=None, causal=True).view(
            bsz, q_len + past_kv_len, -1
        )
        #0.0是注意力缩放分数，-1是在自动计算后面的维度
    else:
        q, indices, cu_q_lens, max_s = unpad_input(q, key_padding_mask)
        # We can skip concat and call unpad twice but seems better to call unpad only once.
        kv, _, cu_k_lens, max_k = unpad_input(
            torch.stack((k, v), dim=2), key_padding_mask
        )#沿第二维度堆叠，kv：去除了填充的 k 和 v 张量。_：这个下划线 _ 是一个占位符，表示这部分的输出我们不关心，所以没有给变量命名。
        #cu_k_lens：累积长度，表示每个序列的长度。max_k：最大长度，表示所有序列中最长的长度。
        output_unpad = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_q_lens,
            cu_k_lens,
            max_s,
            max_k,
            0.0,
            softmax_scale=None,
            causal=True,
        )
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len + past_kv_len)

    # Need to strip off the zero query outputs.#如果存在过去的键值对，那么这段代码会更新 output 变量，移除前面全0的部分，哪一步实现了？
    if past_kv_len > 0:
        output = output[:, past_kv_len:, ...]
#这段代码的目的是为了在自注意力计算中，如果模型之前已经计算过一些键值对，那么在新的计算中就不再考虑这些旧的键值对，而是只关注新的输入。
    return self.o_proj(output), None, past_key_value


# Disable the transformation of the attention mask in LlamaModel as flash attention
#决定禁用对注意力掩码（attention mask）的转换，因为flash attention机制需要的是一个布尔类型的键填充掩码（key_padding_mask）。这里的“禁用转换”指的是不进行一些可能在其他情况下会进行的注意力掩码的变换操作。？？？什么意思
# takes a boolean key_padding_mask. Fills in the past kv length for use in forward.
def _prepare_decoder_attention_mask(
    """这个函数的作用是在进行自注意力计算时，如果模型之前已经计算过一些键值对（即存在 past_key_values_length），
    那么需要在注意力掩码中加入这些已经计算过的部分，以确保模型在新的计算中不再考虑这些旧的键值对。"""
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            (#torch.full 是 PyTorch 库中的一个函数，它用于创建一个给定形状（shape）和数据类型（dtype）的张量（tensor），并用指定的值填充。
                #这个函数的主要用途是生成一个所有元素都是相同值的张量，类似于 NumPy 中的 np.full 函数。
                torch.full(
                    (input_shape[0], past_key_values_length),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ),
            dim=-1,
        )

    if attention_mask is not None and torch.all(attention_mask):#如果attention_mask全是True，说明信息全是有用的
        return None  # This uses the faster call when training with full samples

    return attention_mask


def replace_llama_attn_with_flash_attn():
    #这个flash-attention好像用不了，对cuda版本有要求，只能支持A100 or H100 GPU
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )

    LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    LlamaAttention.forward = forward


def test():
    from fastchat.train.llama_flash_attn_monkey_patch import forward as fastchat_forward
    from transformers.models.llama.configuration_llama import LlamaConfig

    config = LlamaConfig(
        hidden_size=1024,
        intermediate_size=128,#这个参数可能用于控制某个算法或者程序中的中间状态的大小，
        num_hidden_layers=1,
        num_attention_heads=8,
        max_position_embeddings=16,
    )
    device = torch.device("cuda")
    model = LlamaModel(config)
    attn = LlamaAttention(config).to(device).half()#.half()：这个方法将模型的权重和输入数据转换为半精度浮点数（FP16）。这样做可以减少模型的内存占用，并可能提高计算速度，尤其是在支持FP16运算的GPU上。
    bsz, hs, seqlen = 2, config.hidden_size, config.max_position_embeddings
    position_ids = torch.arange(seqlen, dtype=torch.long, device=device).view(
        -1, seqlen
    )

    mask = torch.full((bsz, seqlen), True, dtype=torch.bool, device=device)#形状为(bsz, seqlen)值为True类型 dtype=torch.bool
    for i in range(4):
        hidden = torch.rand((bsz, seqlen, hs), dtype=torch.float16, device=device)#hs表示隐藏层的大小
        if i:
            mask[0, -i:] = False#表示在第一个序列（mask 的第一行）的最后 i 个位置设置为 False，意味着这些位置在计算注意力时将被忽略。
            mask[1, :i] = False
        #调用元原模型的函数为解码器编写掩码
        lmask = model._prepare_decoder_attention_mask(mask, hidden.shape[:2], hidden, 0)
        #使用解码器的attention计算注意力分数结果
        ref, _, _ = attn.forward(
            hidden, attention_mask=lmask, position_ids=position_ids
        )
        #使用fastchat_forward计算attention，不使用解码器的attention
        fast, _, _ = fastchat_forward(
            attn, hidden, attention_mask=mask, position_ids=position_ids
        )
        #使用一个名为LlamaModel的模型，并且对其进行了某种形式的“猴子补丁”（monkey patching），即动态地修改了模型的某些方法。这是作者刚才设计的attention
        lmask = _prepare_decoder_attention_mask(
            model, mask, hidden.shape[:2], hidden, 0
        )
        test, _, _ = forward(
            attn, hidden, attention_mask=lmask, position_ids=position_ids
        )
        #对三种attention模型的结果进行对比
        print(f"Mean(abs(ref)) = {torch.mean(torch.abs(ref))}")
        print(f"Mean(abs(ref - fast)) = {torch.mean(torch.abs(ref - fast))}")
        print(f"Mean(abs(ref - test)) = {torch.mean(torch.abs(ref - test))}")
        print(f"Mean(abs(fast - test)) = {torch.mean(torch.abs(fast - test))}")
        print(f"allclose(fast, test) = {torch.allclose(fast, test)}")

    with torch.no_grad():
        # Also check that past_kv is handled properly
        #同时验证了一次性处理整个序列和分apart处理序列的结果是否一致
        hidden = torch.rand((bsz, seqlen, hs), dtype=torch.float16, device=device)
        part_len = seqlen // 4
        assert part_len * 4 == seqlen
        mask = torch.full((bsz, seqlen), True, dtype=torch.bool, device=device)
        mask[0, -2:] = False#这行代码将mask张量中第一行的最后两个元素设置为False，表示这两个位置是无效的。
        lmask = _prepare_decoder_attention_mask(
            model, mask, hidden.shape[:2], hidden, 0
        )
        oneshot, _, _ = forward(
            attn, hidden, attention_mask=lmask, position_ids=position_ids
        )
        parts = []
        past_kv, past_kv_len = None, 0
        for i in range(4):
            start = part_len * i
            end = start + part_len
            hidden_part = hidden[:, start:end, ...]
            lmask = _prepare_decoder_attention_mask(
                model,
                mask[:, start:end],
                hidden_part.shape[:2],
                hidden_part,
                past_kv_len,
            )
            part, _, past_kv = forward(
                attn,
                hidden_part.clone(),
                attention_mask=lmask,
                position_ids=position_ids[:, start:end],
                past_key_value=past_kv,
                use_cache=True,
            )
            parts.append(part)
            past_kv_len = past_kv[0].shape[1]

        print(
            f"allclose(oneshot[:, 0], parts[0]) = {torch.allclose(oneshot[:, :part_len], parts[0])}"
        )
        print(
            f"allclose(oneshot, parts) = {torch.allclose(oneshot, torch.cat(parts, dim=1))}"
        )
        #torch.allclose函数用于比较两个张量是否在一定的容忍度内相等。如果两个张量在元素级别上的差异不超过指定的容忍度，
        #则认为它们是相等的。这里比较的是oneshot张量的第一列（即整个序列的输出）和分割序列后第一部分的输出

if __name__ == "__main__":
    test()
