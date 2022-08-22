import random
import torch
from torch import nn
from dataset import batchify


class DisLoss(nn.Module):
    def __init__(self, device):
        super(DisLoss, self).__init__()
        self.device = device
        
    def forward(self, emb_batch, embedding, attr_sim, indices, beta, **kwargs):
        loss = torch.autograd.Variable(
            torch.scalar_tensor(0.),
            requires_grad=True
        ).to(self.device)
        for i, emb_i in enumerate(emb_batch):
            a = embedding[indices[i]] - emb_i
            b = torch.sum(a.pow(2), -1)
            c = torch.matmul(b, attr_sim[i].T)
            loss += torch.matmul(
                torch.sum((embedding[indices[i]] - emb_i).pow(2), -1),
                attr_sim[i].T
            )
        return loss / len(emb_batch)


class SimLoss(nn.Module):
    def __init__(self, device):
        super(SimLoss, self).__init__()
        self.device = device

    def forward(self, embedding, sim_x, beta, **kwargs):
        loss = torch.autograd.Variable(
            torch.scalar_tensor(0.),
            requires_grad=True
        ).to(self.device)
        l1 = torch.sqrt(torch.sum(
            torch.pow(embedding, 2),
            dim=1
        ))
        for i, emb_i in enumerate(embedding):
            # a = torch.matmul(embedding, emb_i.T)
            # b = l1 * l1[i]
            # c = a / b
            # d = c * sim_x[i]
            loss += torch.sum(torch.matmul(embedding, emb_i.T) / (l1 * l1[i]) * sim_x[i])
            # loss += embedding / l1 * emb_i.T / l1[i]
        return -loss


def _train_step(lm_model, emb_model, x, attr, y):
    logits, probs = lm_model(emb_model(x, attr))
    logits = logits.view(-1, logits.size(-1))
    probs = probs.view(-1, probs.size(-1))
    y = y.view(-1)
    loss = nn.functional.cross_entropy(logits, y)
    acc = torch.sum(torch.argmax(probs, dim=1) == y).float() / len(y)
    return loss, acc


def clip_grad(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def lm_train_iter(steps, writer, lm_model, emb_model,
                  optimizer, scheduler, batch_size,
                  nb_batches_per_iter, data, attr,
                  train_pos, grad_clip, warmup_steps):
    # data_batch = batchify(data, batch_size)
    data_batch = data
    lm_model.train()
    emb_model.train()
    optimizer.zero_grad()
    loss_all = 0
    acc_all = 0
    for i in range(nb_batches_per_iter):
    # for i in range(0, len(data_batch)):
    #     x = data_batch[i].contiguous().long().cuda()
        x = data_batch[train_pos].contiguous().long().cuda()
        if attr is not None:
            attr_x = attr[x].cuda()
        else:
            attr_x = None
        y = torch.zeros_like(x)
        y[:, :-1] = x[:, 1:]
        y[:, -1] = x[:, -2]

        loss, acc = _train_step(lm_model=lm_model,
                                emb_model=emb_model,
                                x=x,
                                attr=attr_x,
                                y=y)

        loss.backward()
        clip_grad(optimizer, grad_clip)
        optimizer.step()

        loss_all += loss
        acc_all += acc
        steps += 1
        train_pos += 1
        if train_pos > len(data_batch) - 1:
            train_pos = 0

        writer.add_scalar('losses/lm_loss', loss, steps)
        writer.add_scalar('losses/lm_acc', acc, steps)
        writer.add_scalar('lrs/lr_lm', scheduler.get_lr()[0], steps)
        writer.flush()

        if i % 10 == 0:
            print('lm_batch: {}/{}, lm_loss: {:.6f}, lm_acc: {:.4f}, lm_lr: {:.8f}'.format(
                i,
                nb_batches_per_iter,
                loss,
                acc,
                scheduler.get_lr()[0]
            ))
        if steps < warmup_steps:
            scheduler.step()
        elif steps % 10 == 0:
            scheduler.step()
        torch.cuda.empty_cache()
    loss_all = loss_all / nb_batches_per_iter
    acc_all = acc_all / nb_batches_per_iter
    return loss_all, acc_all, train_pos


def dis_train_iter(steps, writer, emb_model, optimizer, scheduler,
                   adj, data, beta, attr,
                   nb_batches_per_iter_max, block_size, device, mode, indices):
    actual_nb_batches_per_iter = 0
    loss_all = 0
    data = data.cuda()
    if attr is not None:
        attr = attr.cuda()
    for i in range(nb_batches_per_iter_max):
    # for i in range(0, len(data), block_size):
        # print(train_pos)
        train_pos = random.sample(list(range(len(data))), block_size)
        # print('random train pos:', train_pos)
        # train_pos = np.arange(i, i + block_size)
        actual_nb_batches_per_iter += 1
        X = data[train_pos]
        # X = data[i * block_size: (i+1)*block_size]
        if attr is not None:
            attr_x = attr[train_pos]
            # attr_x = attr[i:i + block_size]
        else:
            attr_x = None
        # adj_x = adj[train_pos][:, train_pos]
        adj_x = adj[train_pos].cuda()
        # adj_x = adj[i*block_size: (i+1)*block_size, i*block_size: (i+1)*block_size]
        optimizer.zero_grad()
        indices_x = indices[train_pos]

        out_x = emb_model(X, attr_x)
        out = emb_model(data, attr)
        # 优化 generator.output 的参数
        # out_x = emb_model.output.weight[X]
        # out = emb_model.output.weight
        
        if mode == 'dis':
            dis_loss = DisLoss(device)
        elif mode == 'cos':
            dis_loss = SimLoss(device)
        loss = dis_loss(out_x, out,
                        adj_x[torch.arange(0, block_size).long().unsqueeze(1), indices_x],
                        indices_x,
                        beta).to(device)
        loss.backward()
        steps = steps + 1
        optimizer.step()
        # if steps > 1600:
        #     scheduler.step()
        writer.add_scalar('losses/dis_loss', loss, steps)
        writer.flush()
        if i % 10 == 0:
            print('dis_batch: {}/{}, dis_loss: {:.6f}, dis_lr: {:.6f}'.format(i, nb_batches_per_iter_max, loss,
                                                                              scheduler.get_lr()[0]))
        loss_all += loss.item()
        writer.add_scalar('lrs/lr_dis', scheduler.get_lr()[0], steps)
        writer.flush()
        torch.cuda.empty_cache()
    loss_all = loss_all / actual_nb_batches_per_iter
    return loss_all
