from copy import deepcopy
from utils.attention import *

def get_attn_mask(len_q, n_head, seq_k, mask_token_id):
    batch_size, len_k = seq_k.shape
    padMask = seq_k.data.eq(mask_token_id).unsqueeze(1)   # [batch_size, 1, len_k]
    padMask = padMask.expand(batch_size, len_q, len_k)
    padMask = padMask.unsqueeze(1)
    return padMask.expand(batch_size, n_head, len_q, len_k)

def get_attn_subsequence_mask(batch_size, n_head, m, device):
    subsequence_mask = torch.triu(torch.ones((batch_size, n_head, m, m)), 1)
    return subsequence_mask.to(device)

class PostionalEncoding(nn.Module):
    '''
        加上位置信息
    '''
    def __init__(self, d_model, max_len=5000, device="cuda"):
        super(PostionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        div = torch.exp(_2i * -np.log(10000) / d_model)

        self.encoding[:, 0::2] = torch.sin(pos * div)
        self.encoding[:, 1::2] = torch.cos(pos * div)

    def forward(self, x):
        '''
            x: (batch_size, m, d_model)
        '''
        x = self.encoding[:x.size(1), :] + x
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.ln = nn.LayerNorm(d_model)
    def forward(self, input):
        '''
        :param input: [batch_size, m, d_model]
        :return: Add & Layer Norm [batch_size, m, d_model]
        '''
        output = self.fc(input)
        # output = nn.LayerNorm(self.d_model).cuda()(output + input)
        output = self.ln(output + input)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, n_head, device="cuda"):
        '''
        :param d_model: input传入的最后一维，即每个xi的特征长度
        :param o_model: 经过一层 EncoderLayer 后，出来的维度是多少，一般等于d_model
        :param dim_feedforward: 决定 w_o 的dim=1的维度是多少
        :param n_head: 多少个head
        :param device: cuda
        '''
        super(EncoderLayer, self).__init__()
        self.enc_attention = mutliHeadAttention(d_model, dim_feedforward, n_head, device=device)
        self.enc_ffn = FeedForward(d_model).to(device)
    def forward(self, enc_input, encMask):
        '''
        :param enc_input: [batch_size, m, d_model]
        :return: [batch_size, m, o_model]
        '''
        output = self.enc_attention(enc_input, enc_input, enc_input, encMask)
        output = self.enc_ffn(output)
        return output

class Encoder(nn.Module):
    def __init__(self, EncodeLayer, n, word2vec, d_model, mask_token_id, device="cuda"):
        super(Encoder, self).__init__()
        self.word2vec = word2vec
        self.num_head = EncodeLayer.enc_attention.num_head
        self.mask_token_id = mask_token_id
        self.posEmb = PostionalEncoding(d_model, device=device)
        self.layers = nn.ModuleList([deepcopy(EncodeLayer) for _ in range(n)])
    def forward(self, encInput):
        '''
        :param encInput: [batch_size, m]
        :return: [batch_size, m, o_model]
        '''
        _, m = encInput.shape
        encOutput = self.word2vec(encInput)
        encOutput = self.posEmb(encOutput)
        encMask = get_attn_mask(m, self.num_head, encInput, self.mask_token_id)
        for layer in self.layers:
            encOutput = layer(encOutput, encMask)
        return encOutput

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, device="cuda"):
        super(DecoderLayer, self).__init__()
        self.dec_mask_attn = mutliHeadAttention(d_model, d_model, n_head, device=device)
        self.dec_attn = mutliHeadAttention(d_model, d_model, n_head, device=device)
        self.ffn = FeedForward(d_model).to(device)
    def forward(self, encInput, decInput, attnMask, selfMask):
        output = self.dec_mask_attn(decInput, decInput, decInput, selfMask)
        output = self.dec_attn(output, encInput, encInput, attnMask)
        output = self.ffn(output)
        return output

class Decoder(nn.Module):
    def __init__(self, decoderlayer, n, word2vec, d_model, mask_token_id, device="cuda"):
        super(Decoder, self).__init__()
        self.num_head = decoderlayer.dec_attn.num_head
        self.word2vec = word2vec
        self.posEmb = PostionalEncoding(d_model, device=device)
        self.layers = nn.ModuleList([deepcopy(decoderlayer) for _ in range(n)])
        self.mask_token_id = mask_token_id
        self.device = device
    def forward(self, encInput, decInput, encOutput, attnMask=None, encdecMask=None):
        '''
        :param encInput: [batch_size, src_m]
        :param decInput: [batch_size, tar_m]
        :param encOutput: [batch_size, m, d_model]
        :return:
        '''
        batch_size, m = decInput.shape
        attnMask = get_attn_mask(m, self.num_head, decInput, self.mask_token_id)
        decInput = self.word2vec(decInput)
        decInput = self.posEmb(decInput)
        selfMask = get_attn_subsequence_mask(batch_size, self.num_head, m, self.device)
        if attnMask is not None:
            selfMask = torch.gt((attnMask + selfMask), 0).to(self.device)
        encdecMask = get_attn_mask(m, self.num_head, encInput, self.mask_token_id)
        for layer in self.layers:
            decInput = layer(encOutput, decInput, encdecMask, selfMask.to(self.device))
        return decInput

class Transformer(nn.Module):
    def __init__(self, vocab_size, mask_token_id, d_model, n_head, num_of_layer, device="cuda"):
        super(Transformer, self).__init__()
        encodeLayer = EncoderLayer(d_model, d_model, n_head, device)
        decodeLayer = DecoderLayer(d_model, n_head, device)
        embedding = nn.Embedding(vocab_size, d_model).to(device)
        share_weight = embedding.weight
        self.encoder = Encoder(encodeLayer, num_of_layer, embedding, d_model, mask_token_id, device)
        self.decoder = Decoder(decodeLayer, num_of_layer, embedding, d_model, mask_token_id, device)
        self.generator = nn.Linear(d_model, vocab_size, bias=False).to(device)
        self.generator.weight = share_weight
    def forward(self, encInput, decInput):
        '''
        :param encInput: [batch_size, src_m]
        :param decInput: [batch_size, tar_m]
        :return:
        '''
        encOutput = self.encoder(encInput)
        output = self.decoder(encInput, decInput, encOutput)    # [batch_size, tar_m, d_model]
        output = self.generator(output)    # [batch_size, tar_m, tar_vocab_num]
        return output.view(-1, output.size(-1)) # [batch_size*tar_m, tar_vocab_num]

# e.g
# device = "cuda" if torch.cuda.is_available() else "cpu"
# EncodeLayer = EncoderLayer(80, 256, 2)
# e = Encoder(EncodeLayer, 6)
# output = e(x)
