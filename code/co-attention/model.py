from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

eps = 1e-12

use_cuda = torch.cuda.is_available()

def get_mask_from_seq_len(seq_lens_np, max_len=None):
    seq_lens = torch.from_numpy(seq_lens_np)
    if use_cuda:
        seq_lens = seq_lens.cuda()
    if max_len is None:
        max_len = np.max(seq_lens_np)
    indices = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
    if use_cuda:
        indices = indices.cuda()
    mask = Variable((indices < seq_lens.unsqueeze(1)).float())
    if use_cuda:
        mask = mask.cuda()
    return mask

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

class MaxOutHighway(nn.Module):
    def __init__(self, hidden_dim, maxout_pool_size, dropout_ratio):
        super(MaxOutHighway, self).__init__()
        self.hidden_dim = hidden_dim
        self.maxout_pool_size = maxout_pool_size

        self.r = nn.Linear(5 * hidden_dim, hidden_dim, bias=False)
        self.dropout_r = nn.Dropout(p=dropout_ratio)

        self.m_t_1_mxp = nn.Linear(3 * hidden_dim, hidden_dim*maxout_pool_size)
        self.dropout_m_t_1 = nn.Dropout(p=dropout_ratio)

        self.m_t_2_mxp = nn.Linear(hidden_dim, hidden_dim*maxout_pool_size)
        self.dropout_m_t_2 = nn.Dropout(p=dropout_ratio)

        self.m_t_12_mxp = nn.Linear(2 * hidden_dim, maxout_pool_size)

    def forward(self, h_i, U, d_lens, curr_mask, idx_i_1, u_cat, D_mask, mask_mult, target=None):
        b, m, _ = list(U.size())

        r = F.tanh(self.r(torch.cat((h_i.view(-1, self.hidden_dim), u_cat), 1)))  # b x 5l => b x l
        r = self.dropout_r(r)

        r_expanded = r.unsqueeze(1).expand(b, m, self.hidden_dim).contiguous()  # b x m x l

        m_t_1_in = torch.cat((U, r_expanded), 2).view(-1, 3*self.hidden_dim)  # b*m x 3l

        m_t_1 = self.m_t_1_mxp(m_t_1_in)  # b*m x p*l
        m_t_1 = self.dropout_m_t_1(m_t_1)
        m_t_1, _ = m_t_1.view(-1, self.hidden_dim, self.maxout_pool_size).max(2) # b*m x l

        m_t_2 = self.m_t_2_mxp(m_t_1)  # b*m x l*p
        m_t_2 = self.dropout_m_t_2(m_t_2)
        m_t_2, _ = m_t_2.view(-1, self.hidden_dim, self.maxout_pool_size).max(2)  # b*m x l

        alpha_in = torch.cat((m_t_1, m_t_2), 1)  # b*m x 2l
        alpha = self.m_t_12_mxp(alpha_in)  # b * m x p
        alpha, _ = alpha.max(1)  # b*m
        alpha = alpha.view(-1, m) # b x m

        alpha = alpha * D_mask + mask_mult  # b x m
        alpha = F.log_softmax(alpha, 1)  # b x m
        _, idx_i = torch.max(alpha, dim=1)
        ''' 
        d_lens_var = Variable(torch.from_numpy(d_lens))
        if use_cuda:
            d_lens_var = d_lens_var.cuda()

        idx_i = torch.min(idx_i, d_lens_var)
        '''

        # ??both start and end should be same or treat them individually
        if curr_mask is None:
            curr_mask = (idx_i == idx_i) # all one #(idx_i != idx_i_1)
        else:
            idx_i = idx_i*curr_mask.long()
            idx_i_1 = idx_i_1*curr_mask.long()
            curr_mask = (idx_i != idx_i_1)

        if target is not None:
            step_loss = -torch.gather(alpha, 1, target.unsqueeze(1)).squeeze()
            step_loss = step_loss * curr_mask.float()

            return idx_i, curr_mask, step_loss
        else:
            return idx_i, curr_mask

class CoattentionModel(nn.Module):
    def __init__(self, hidden_dim, maxout_pool_size, emb_matrix, max_dec_steps, dropout_ratio):
        super(CoattentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_dec_steps = max_dec_steps

        self.embedding = get_pretrained_embedding(emb_matrix)
        self.emb_dim = self.embedding.embedding_dim

        self.dropout_embd = nn.Dropout(p=dropout_ratio)

        self.encoder = nn.LSTM(self.emb_dim, hidden_dim, 1, batch_first=True, bidirectional=False)
        init_lstm_forget_bias(self.encoder)
        #?? should sentinel be shared
        self.d_sentinel = nn.Parameter(torch.rand(hidden_dim,))
        self.q_sentinel = nn.Parameter(torch.rand(hidden_dim,))

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)

        self.fusion_bilstm = nn.LSTM(3*hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=True)
        init_lstm_forget_bias(self.fusion_bilstm)

        self.decoder = nn.LSTM(4*hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=False)
        init_lstm_forget_bias(self.decoder)

        self.maxout_start = MaxOutHighway(hidden_dim, maxout_pool_size, dropout_ratio)
        self.maxout_end = MaxOutHighway(hidden_dim, maxout_pool_size, dropout_ratio)

    def forward(self, q_seq, q_lens, d_seq, d_lens, span=None):
        #document processing
        d_seq_embd = self.embedding(d_seq)
        #d_seq_embd = self.dropout_embd(d_seq_embd)
        ctx_packed = pack_padded_sequence(d_seq_embd, d_lens, batch_first=True)
        ctx_output, _ = self.encoder(ctx_packed)
        e_ctx, _ = pad_packed_sequence(ctx_output, batch_first=True)
        D = e_ctx.contiguous()  # B x m x l
        b, m, _ = list(D.size())

        # copy sentinel vector at the end
        d_sentinel_exp = self.d_sentinel.unsqueeze(0).expand(b, self.hidden_dim).unsqueeze(1).contiguous()  # B x 1 x l
        indices = torch.from_numpy(d_lens)
        if use_cuda:
            indices = indices.cuda()
        indices = indices.unsqueeze(1).expand(b, self.hidden_dim).unsqueeze(1)

        sentinel_zero = torch.zeros(b, 1, self.hidden_dim)
        if use_cuda:
            sentinel_zero = sentinel_zero.cuda()
        D = torch.cat([D, sentinel_zero], 1)  # B x m + 1 x l
        D = D.scatter_(1, indices, d_sentinel_exp)

        # query processing
        q_lens_idx = torch.from_numpy(np.ascontiguousarray(np.flip(np.argsort(q_lens), axis=0)))
        if use_cuda:
            q_lens_idx = q_lens_idx.cuda()
        q_lens_idx_rev = torch.from_numpy(np.ascontiguousarray(np.argsort(q_lens_idx)))
        if use_cuda:
            q_lens_idx_rev = q_lens_idx_rev.cuda()
        q_lens_ = q_lens[q_lens_idx]

        q_seq_ = torch.index_select(q_seq, 0, q_lens_idx)
        q_seq_embd = self.embedding(q_seq_)
        #q_seq_embd = self.dropout_embd(q_seq_embd)

        q_packed = pack_padded_sequence(q_seq_embd, q_lens_, batch_first=True)
        q_output, _ = self.encoder(q_packed)
        e_q_, _ = pad_packed_sequence(q_output, batch_first=True)
        e_q_ = e_q_.contiguous()

        e_q = torch.index_select(e_q_, 0, q_lens_idx_rev) #B x n x l

        #copy sentinel vector at the end
        q_sentinel_exp = self.q_sentinel.unsqueeze(0).expand(b, self.hidden_dim).unsqueeze(1).contiguous() #B x 1 x l
        indices = torch.from_numpy(q_lens)
        if use_cuda:
            indices = indices.cuda()
        indices = indices.unsqueeze(1).expand(b, self.hidden_dim).unsqueeze(1)
        sentinel_zero = torch.zeros(b, 1, self.hidden_dim)
        if use_cuda:
            sentinel_zero = sentinel_zero.cuda()
        e_q = torch.cat([e_q, sentinel_zero], 1) #B x n + 1 x l
        e_q = e_q.scatter_(1, indices, q_sentinel_exp)

        #project q
        Q = F.tanh(self.q_proj(e_q.view(-1, self.hidden_dim))).view(e_q.size()) #B x n + 1 x l

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

        bilstm_in = torch.cat((C_D_t, D), 2) # B x m + 1 x 3l

        #?? should it be d_lens + 1 and get U[:-1]
        bilstm_in_packed = pack_padded_sequence(bilstm_in, d_lens, batch_first=True)
        bilstm_in_output, _ = self.fusion_bilstm(bilstm_in_packed)
        U, _ = pad_packed_sequence(bilstm_in_output, batch_first=True)
        U = U.contiguous() #B x m x 2l

        #??how to initialize s_i_1, e_i_1
        s_i_1 = Variable(torch.zeros(b)).long()
        e_i_1 = Variable(torch.zeros(b).fill_(m-1)).long()
        if use_cuda:
            s_i_1 = s_i_1.cuda()
            e_i_1 = e_i_1.cuda()

        dec_state_i = None
        step_losses = []
        curr_mask_s = None
        curr_mask_e = None

        results_mask_s = []
        results_s = []

        results_mask_e = []
        results_e = []

        D_mask = get_mask_from_seq_len(d_lens)
        mask_mult = (1.0 - D_mask) * (-1e30)

        for _ in range(self.max_dec_steps):
            indices = torch.arange(0, b, out=torch.LongTensor(b))
            u_s_i_1 = U[indices, s_i_1, :] #b x 2l
            u_e_i_1 = U[indices, e_i_1, :] #b x 2l
            u_cat = torch.cat((u_s_i_1, u_e_i_1), 1) #b x 4l
            lstm_out, dec_state_i = self.decoder(u_cat.unsqueeze(1), dec_state_i)
            h_i, c_i = dec_state_i

            if span is not None:
                s_i_1, curr_mask_s, step_loss_s = self.maxout_start(h_i, U, d_lens, curr_mask_s, s_i_1, u_cat, D_mask, mask_mult, span[:, 0])
                e_i_1, curr_mask_e, step_loss_e = self.maxout_end(h_i, U, d_lens, curr_mask_e, e_i_1, u_cat, D_mask, mask_mult, span[:, 1])
                step_loss = step_loss_s + step_loss_e
                step_losses.append(step_loss)
            else:
                s_i_1, curr_mask_s = self.maxout_start(h_i, U, d_lens, curr_mask_s, s_i_1, u_cat, D_mask, mask_mult)
                e_i_1, curr_mask_e = self.maxout_end(h_i, U, d_lens, curr_mask_e, e_i_1, u_cat, D_mask, mask_mult)

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

        if span is not None:
            sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
            batch_avg_loss = sum_losses / self.max_dec_steps
            loss = torch.mean(batch_avg_loss)
            return loss, idx_s, idx_e
        else:
            return idx_s, idx_e

