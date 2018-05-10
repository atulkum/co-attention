from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

use_cuda = torch.cuda.is_available()

#out-of-vocabulary words to zero
def get_pretrained_embedding(np_embd):
    embedding = nn.Embedding(*np_embd.shape)
    embedding.weight = nn.Parameter(torch.from_numpy(np_embd).float())
    embedding.weight.requires_grad = False
    return embedding

def init_lstm_forget_bias(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

class Encoder(nn.Module):
    def __init__(self, hidden_dim, emb_matrix, dropout_ratio):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = get_pretrained_embedding(emb_matrix)
        self.emb_dim = self.embedding.embedding_dim

        self.encoder = nn.LSTM(self.emb_dim, hidden_dim, 1, batch_first=True,
                              bidirectional=False, dropout=dropout_ratio)
        init_lstm_forget_bias(self.encoder)
        self.dropout_emb = nn.Dropout(p=dropout_ratio)
        self.sentinel = nn.Parameter(torch.rand(hidden_dim,))

    def forward(self, seq, mask):
        lens = torch.sum(mask, 1)
        lens_sorted, lens_argsort = torch.sort(lens, 0, True)
        _, lens_argsort_argsort = torch.sort(lens_argsort, 0)
        seq_ = torch.index_select(seq, 0, lens_argsort)

        seq_embd = self.embedding(seq_)
        packed = pack_padded_sequence(seq_embd, lens_sorted, batch_first=True)
        output, _ = self.encoder(packed)
        e, _ = pad_packed_sequence(output, batch_first=True)
        e = e.contiguous()
        e = torch.index_select(e, 0, lens_argsort_argsort)  # B x m x 2l
        e = self.dropout_emb(e)

        b, _ = list(mask.size())
        # copy sentinel vector at the end
        sentinel_exp = self.sentinel.unsqueeze(0).expand(b, self.hidden_dim).unsqueeze(1).contiguous()  # B x 1 x l
        lens = lens.unsqueeze(1).expand(b, self.hidden_dim).unsqueeze(1)

        sentinel_zero = torch.zeros(b, 1, self.hidden_dim)
        if use_cuda:
            sentinel_zero = sentinel_zero.cuda()
        e = torch.cat([e, sentinel_zero], 1)  # B x m + 1 x l
        e = e.scatter_(1, lens, sentinel_exp)

        return e

class FusionBiLSTM(nn.Module):
    def __init__(self, hidden_dim, dropout_ratio):
        super(FusionBiLSTM, self).__init__()
        self.fusion_bilstm = nn.LSTM(3 * hidden_dim, hidden_dim, 1, batch_first=True,
                                     bidirectional=True, dropout=dropout_ratio)
        init_lstm_forget_bias(self.fusion_bilstm)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, seq, mask):
        lens = torch.sum(mask, 1)
        lens_sorted, lens_argsort = torch.sort(lens, 0, True)
        _, lens_argsort_argsort = torch.sort(lens_argsort, 0)
        seq_ = torch.index_select(seq, 0, lens_argsort)
        packed = pack_padded_sequence(seq_, lens_sorted, batch_first=True)
        output, _ = self.fusion_bilstm(packed)
        e, _ = pad_packed_sequence(output, batch_first=True)
        e = e.contiguous()
        e = torch.index_select(e, 0, lens_argsort_argsort)  # B x m x 2l
        e = self.dropout(e)
        return e

class DynamicDecoder(nn.Module):
    def __init__(self, hidden_dim, maxout_pool_size, max_dec_steps, dropout_ratio):
        super(DynamicDecoder, self).__init__()
        self.max_dec_steps = max_dec_steps
        self.decoder = nn.LSTM(4 * hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=False)
        init_lstm_forget_bias(self.decoder)

        self.maxout_start = MaxOutHighway(hidden_dim, maxout_pool_size, dropout_ratio)
        self.maxout_end = MaxOutHighway(hidden_dim, maxout_pool_size, dropout_ratio)

    def forward(self, U, d_mask, span):
        b, m, _ = list(U.size())

        curr_mask_s,  curr_mask_e = None, None
        results_mask_s, results_s = [], []
        results_mask_e, results_e = [], []
        step_losses = []

        # decoder
        mask_mult = (1.0 - d_mask.float()) * (-1e30)

        indices = torch.arange(0, b, out=torch.LongTensor(b))

        # ??how to initialize s_i_1, e_i_1
        s_i_1 = torch.zeros(b, ).long()
        e_i_1 = torch.sum(d_mask, 1)
        e_i_1 = e_i_1 - 1

        if use_cuda:
            s_i_1 = s_i_1.cuda()
            e_i_1 = e_i_1.cuda()
            indices = indices.cuda()

        dec_state_i = None

        for _ in range(self.max_dec_steps):
            u_s_i_1 = U[indices, s_i_1, :]  # b x 2l
            u_e_i_1 = U[indices, e_i_1, :]  # b x 2l
            u_cat = torch.cat((u_s_i_1, u_e_i_1), 1)  # b x 4l

            lstm_out, dec_state_i = self.decoder(u_cat.unsqueeze(1), dec_state_i)
            h_i, c_i = dec_state_i
            s_target = None
            e_target = None

            if span is not None:
                s_target = span[:, 0]
                e_target = span[:, 1]

            s_i_1, curr_mask_s, step_loss_s = self.maxout_start(h_i, U, curr_mask_s, s_i_1,
                                                                u_cat, mask_mult, s_target)
            e_i_1, curr_mask_e, step_loss_e = self.maxout_end(h_i, U, curr_mask_e, e_i_1,
                                                              u_cat, mask_mult, e_target)

            if span is not None:
                step_loss = step_loss_s + step_loss_e
                step_losses.append(step_loss)

            results_mask_s.append(curr_mask_s)
            results_s.append(s_i_1)
            results_mask_e.append(curr_mask_e)
            results_e.append(e_i_1)

        result_pos_s = torch.sum(torch.stack(results_mask_s, 1), 1).long()
        result_pos_s = result_pos_s - 1
        idx_s = torch.gather(torch.stack(results_s, 1), 1, result_pos_s.unsqueeze(1)).squeeze()

        result_pos_e = torch.sum(torch.stack(results_mask_e, 1), 1).long()
        result_pos_e = result_pos_e - 1
        idx_e = torch.gather(torch.stack(results_e, 1), 1, result_pos_e.unsqueeze(1)).squeeze()

        loss = None

        if span is not None:
            sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
            batch_avg_loss = sum_losses / self.max_dec_steps
            loss = torch.mean(batch_avg_loss)

        return loss, idx_s, idx_e


class MaxOutHighway(nn.Module):
    def __init__(self, hidden_dim, maxout_pool_size, dropout_ratio):
        super(MaxOutHighway, self).__init__()
        self.hidden_dim = hidden_dim
        self.maxout_pool_size = maxout_pool_size

        self.r = nn.Linear(5 * hidden_dim, hidden_dim, bias=False)
        #self.dropout_r = nn.Dropout(p=dropout_ratio)

        self.m_t_1_mxp = nn.Linear(3 * hidden_dim, hidden_dim*maxout_pool_size)
        #self.dropout_m_t_1 = nn.Dropout(p=dropout_ratio)

        self.m_t_2_mxp = nn.Linear(hidden_dim, hidden_dim*maxout_pool_size)
        #self.dropout_m_t_2 = nn.Dropout(p=dropout_ratio)

        self.m_t_12_mxp = nn.Linear(2 * hidden_dim, maxout_pool_size)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, h_i, U, curr_mask, idx_i_1, u_cat, mask_mult, target=None):
        b, m, _ = list(U.size())

        r = F.tanh(self.r(torch.cat((h_i.view(-1, self.hidden_dim), u_cat), 1)))  # b x 5l => b x l
        #r = self.dropout_r(r)

        r_expanded = r.unsqueeze(1).expand(b, m, self.hidden_dim).contiguous()  # b x m x l

        m_t_1_in = torch.cat((U, r_expanded), 2).view(-1, 3*self.hidden_dim)  # b*m x 3l

        m_t_1 = self.m_t_1_mxp(m_t_1_in)  # b*m x p*l
        #m_t_1 = self.dropout_m_t_1(m_t_1)
        m_t_1, _ = m_t_1.view(-1, self.hidden_dim, self.maxout_pool_size).max(2) # b*m x l

        m_t_2 = self.m_t_2_mxp(m_t_1)  # b*m x l*p
        #m_t_2 = self.dropout_m_t_2(m_t_2)
        m_t_2, _ = m_t_2.view(-1, self.hidden_dim, self.maxout_pool_size).max(2)  # b*m x l

        alpha_in = torch.cat((m_t_1, m_t_2), 1)  # b*m x 2l
        alpha = self.m_t_12_mxp(alpha_in)  # b * m x p
        alpha, _ = alpha.max(1)  # b*m
        alpha = alpha.view(-1, m) # b x m

        alpha = alpha + mask_mult  # b x m
        alpha = F.log_softmax(alpha, 1)  # b x m
        _, idx_i = torch.max(alpha, dim=1)

        if curr_mask is None:
            curr_mask = (idx_i == idx_i)
        else:
            idx_i = idx_i*curr_mask.long()
            idx_i_1 = idx_i_1*curr_mask.long()
            curr_mask = (idx_i != idx_i_1)

        step_loss = None

        if target is not None:
            step_loss = self.loss(alpha, target)
            step_loss = step_loss * curr_mask.float()

        return idx_i, curr_mask, step_loss


class CoattentionModel(nn.Module):
    def __init__(self, hidden_dim, maxout_pool_size, emb_matrix, max_dec_steps, dropout_ratio):
        super(CoattentionModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(hidden_dim, emb_matrix, dropout_ratio)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_bilstm = FusionBiLSTM(hidden_dim, dropout_ratio)
        self.decoder = DynamicDecoder(hidden_dim, maxout_pool_size, max_dec_steps, dropout_ratio)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, q_seq, q_mask, d_seq, d_mask, span=None):
        Q = self.encoder(q_seq, q_mask) # b x n + 1 x l
        D = self.encoder(d_seq, d_mask)  # B x m + 1 x l

        #project q
        Q = F.tanh(self.q_proj(Q.view(-1, self.hidden_dim))).view(Q.size()) #B x n + 1 x l

        #co attention
        D_t = torch.transpose(D, 1, 2) #B x l x m + 1
        L = torch.bmm(Q, D_t) # L = B x n + 1 x m + 1

        A_Q_ = F.softmax(L, dim=1) # B x n + 1 x m + 1
        A_Q = torch.transpose(A_Q_, 1, 2) # B x m + 1 x n + 1
        C_Q = torch.bmm(D_t, A_Q) # (B x l x m + 1) x (B x m x n + 1) => B x l x n + 1

        Q_t = torch.transpose(Q, 1, 2)  # B x l x n + 1
        A_D = F.softmax(L, dim=2)  # B x n + 1 x m + 1
        C_D = torch.bmm(torch.cat((Q_t, C_Q), 1), A_D) # (B x l x n+1 ; B x l x n+1) x (B x n +1x m+1) => B x 2l x m + 1

        C_D_t = torch.transpose(C_D, 1, 2)  # B x m + 1 x 2l

        #fusion BiLSTM
        bilstm_in = torch.cat((C_D_t, D), 2) # B x m + 1 x 3l
        bilstm_in = self.dropout(bilstm_in)
        #?? should it be d_lens + 1 and get U[:-1]
        U = self.fusion_bilstm(bilstm_in, d_mask) #B x m x 2l

        loss, idx_s, idx_e = self.decoder(U, d_mask, span)
        if span is not None:
            return loss, idx_s, idx_e
        else:
            return idx_s, idx_e
