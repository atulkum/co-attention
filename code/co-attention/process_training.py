import io
import json
import logging
import os
import sys
import time

import numpy as np
import tensorflow as tf
import torch
from data_util.data_batcher import get_batch_generator
from data_util.evaluate import exact_match_score, f1_score
from data_util.official_eval_helper import get_json_data, generate_answers
from data_util.pretty_print import print_example
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from data_util.vocab import get_glove

from config import config
from model import CoattentionModel
from model_baseline import Baseline

logging.basicConfig(level=logging.INFO)

use_cuda = torch.cuda.is_available()

class Processor(object):
    def __init__(self):
        self.glove_path = os.path.join(config.data_dir, "glove.6B.{}d.txt".format(config.embedding_size))
        self.emb_matrix, self.word2id, self.id2word = get_glove(self.glove_path, config.embedding_size)

        self.train_context_path = os.path.join(config.data_dir, "train.context")
        self.train_qn_path = os.path.join(config.data_dir, "train.question")
        self.train_ans_path = os.path.join(config.data_dir, "train.span")
        self.dev_context_path = os.path.join(config.data_dir, "dev.context")
        self.dev_qn_path = os.path.join(config.data_dir, "dev.question")
        self.dev_ans_path = os.path.join(config.data_dir, "dev.span")

    def get_mask_from_seq_len(self, seq_mask):
        seq_lens = np.sum(seq_mask, 1)
        max_len = np.max(seq_lens)
        indices = np.arange(0, max_len)
        mask = (indices < np.expand_dims(seq_lens, 1)).astype(int)
        return mask

    def get_data(self, batch, is_train=True):
        qn_mask = self.get_mask_from_seq_len(batch.qn_mask)
        qn_mask_var = torch.from_numpy(qn_mask).long()

        context_mask = self.get_mask_from_seq_len(batch.context_mask)
        context_mask_var = torch.from_numpy(context_mask).long()

        qn_seq_var = torch.from_numpy(batch.qn_ids).long()
        context_seq_var = torch.from_numpy(batch.context_ids).long()

        if is_train:
            span_var = torch.from_numpy(batch.ans_span).long()

        if use_cuda:
            qn_mask_var = qn_mask_var.cuda()
            context_mask_var = context_mask_var.cuda()
            qn_seq_var = qn_seq_var.cuda()
            context_seq_var = context_seq_var.cuda()
            if is_train:
                span_var = span_var.cuda()

        if is_train:
            return qn_seq_var, qn_mask_var, context_seq_var, context_mask_var, span_var
        else:
            return qn_seq_var, qn_mask_var, context_seq_var, context_mask_var

    def save_model(self, model, optimizer, loss, global_step, epoch, model_dir):
        model_state = model.state_dict()
        model_state = {k: v for k, v in model_state.items() if 'embedding' not in k}

        state = {
            'global_step': global_step,
            'epoch': epoch,
            'model': model_state,
            'optimizer': optimizer.state_dict(),
            'current_loss': loss
        }
        model_save_path = os.path.join(model_dir, 'model_%d_%d_%d' % (global_step, epoch, int(time.time())))
        torch.save(state, model_save_path)

    def get_model(self, model_file_path=None, is_eval=False):
        if config.model_type == 'co-attention':
            model = CoattentionModel(config.hidden_dim, config.maxout_pool_size,
                                 self.emb_matrix, config.max_dec_steps, config.dropout_ratio)
        else:
            model = Baseline(config.hidden_dim, self.emb_matrix, config.dropout_ratio)

        if is_eval:
            model = model.eval()
        if use_cuda:
            model = model.cuda()

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            model.load_state_dict(state['model'], strict=False)

        return model
    def get_grad_norm(self, parameters, norm_type=2):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        return total_norm

    def get_param_norm(self, parameters, norm_type=2):
        total_norm = 0
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        return total_norm

    def train_one_batch(self, batch, model, optimizer, params):
        model.train()
        optimizer.zero_grad()
        q_seq, q_lens, d_seq, d_lens, span = self.get_data(batch)
        loss, _, _ = model(q_seq, q_lens, d_seq, d_lens, span)
        loss.backward()

        param_norm = self.get_param_norm(params)
        grad_norm = self.get_grad_norm(params)

        clip_grad_norm_(params, config.max_grad_norm)
        optimizer.step()

        return loss.item(), param_norm, grad_norm

    def eval_one_batch(self, batch, model):
        model.eval()
        q_seq, q_lens, d_seq, d_lens, span = self.get_data(batch)
        loss, pred_start_pos, pred_end_pos = model(q_seq, q_lens, d_seq, d_lens, span)
        return loss.item(), pred_start_pos.data, pred_end_pos.data

    def test_one_batch(self, batch, model):
        model.eval()
        q_seq, q_lens, d_seq, d_lens = self.get_data(batch, is_train=False)
        pred_start_pos, pred_end_pos = model(q_seq, q_lens, d_seq, d_lens)
        return pred_start_pos.data, pred_end_pos.data

    def train(self, model_file_path):
        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        bestmodel_dir = os.path.join(train_dir, 'bestmodel')
        if not os.path.exists(bestmodel_dir):
           os.makedirs(bestmodel_dir)

        summary_writer = tf.summary.FileWriter(train_dir)

        with open(os.path.join(train_dir, "flags.json"), 'w') as fout:
            json.dump(vars(config), fout)

        model = self.get_model(model_file_path)
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Adam(params, lr=config.lr)

        num_params = sum(p.numel() for p in params)
        logging.info("Number of params: %d" % num_params)

        exp_loss, best_dev_f1, best_dev_em = None, None, None

        epoch = 0
        global_step = 0

        logging.info("Beginning training loop...")
        while config.num_epochs == 0 or epoch < config.num_epochs:
            epoch += 1
            epoch_tic = time.time()
            for batch in get_batch_generator(self.word2id, self.train_context_path,
                                             self.train_qn_path, self.train_ans_path,
                                             config.batch_size, context_len=config.context_len,
                                             question_len=config.question_len, discard_long=True):
                global_step += 1
                iter_tic = time.time()

                loss, param_norm, grad_norm = self.train_one_batch(batch, model, optimizer, params)
                write_summary(loss, "train/loss", summary_writer, global_step)

                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                if not exp_loss:
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                if global_step % config.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))


                if global_step % config.save_every == 0:
                    logging.info("Saving to %s..." % model_dir)
                    self.save_model(model, optimizer, loss, global_step, epoch, model_dir)

                if global_step % config.eval_every == 0:
                    dev_loss = self.get_dev_loss(model)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)

                    train_f1, train_em = self.check_f1_em(model, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (
                        epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)

                    dev_f1, dev_em = self.check_f1_em(model, "dev", num_samples=0)
                    logging.info(
                        "Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)

                    if best_dev_f1 is None or dev_f1 > best_dev_f1:
                        best_dev_f1 = dev_f1

                    if best_dev_em is None or dev_em > best_dev_em:
                        best_dev_em = dev_em
                        logging.info("Saving to %s..." % bestmodel_dir)
                        self.save_model(model, optimizer, loss, global_step, epoch, bestmodel_dir)


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc - epoch_tic))

        sys.stdout.flush()

    def check_f1_em(self, model, dataset, num_samples=100, print_to_screen=False):
        logging.info("Calculating F1/EM for %s examples in %s set..." % (str(num_samples) if num_samples != 0 else "all", dataset))

        if dataset == "train":
            context_path, qn_path, ans_path = self.train_context_path, self.train_qn_path, self.train_ans_path
        elif dataset == "dev":
            context_path, qn_path, ans_path = self.dev_context_path, self.dev_qn_path, self.dev_ans_path
        else:
            raise ('dataset is not defined')

        f1_total = 0.
        em_total = 0.
        example_num = 0

        tic = time.time()

        for batch in get_batch_generator(self.word2id, context_path, qn_path, ans_path, config.batch_size,
                                         context_len=config.context_len, question_len=config.question_len,
                                         discard_long=False):

            pred_start_pos, pred_end_pos = self.test_one_batch(batch, model)

            pred_start_pos = pred_start_pos.tolist()
            pred_end_pos = pred_end_pos.tolist()

            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) \
                    in enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                example_num += 1
                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end + 1]
                pred_answer = " ".join(pred_ans_tokens)

                true_answer = " ".join(true_ans_tokens)

                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)
                f1_total += f1
                em_total += em

                if print_to_screen:
                    print_example(self.word2id, batch.context_tokens[ex_idx], batch.qn_tokens[ex_idx],
                                  batch.ans_span[ex_idx, 0], batch.ans_span[ex_idx, 1], pred_ans_start,
                                  pred_ans_end, true_answer, pred_answer, f1, em)

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        f1_total /= example_num
        em_total /= example_num

        toc = time.time()
        logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))

        return f1_total, em_total

    def get_dev_loss(self, model):
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []
        i = 0
        for batch in get_batch_generator(self.word2id, self.dev_context_path, self.dev_qn_path, self.dev_ans_path,
                                         config.batch_size, context_len=config.context_len,
                                         question_len=config.question_len, discard_long=True):

            loss, _, _ = self.eval_one_batch(batch, model)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)
            i += 1
            if i == 10:
                break
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print "Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic)

        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss

def write_summary(value, tag, summary_writer, global_step):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)

if __name__ == "__main__":
    mode = 'train' #sys.argv[1]
    processor = Processor()
    if mode == "train":
        model_file_path = None
        if len(sys.argv) > 2:
            model_file_path = sys.argv[2]
        processor.train(model_file_path)
    elif mode == "show_examples":
        model_file_path = sys.argv[2]
        model = processor.get_model(model_file_path)
        processor.check_f1_em(model, num_samples=10, print_to_screen=True)

    elif mode == "official_eval":
        model_file_path = sys.argv[2]
        json_in_path = ""
        json_out_path = ""
        qn_uuid_data, context_token_data, qn_token_data = get_json_data(json_in_path)
        model = processor.get_model(model_file_path)
        answers_dict = generate_answers(config, model, processor, qn_uuid_data, context_token_data, qn_token_data)
        print "Writing predictions to %s..." % json_out_path
        with io.open(json_out_path, 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers_dict, ensure_ascii=False)))
            print "Wrote predictions to %s" % json_out_path
