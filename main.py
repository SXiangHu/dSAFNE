import torch
import numpy as np
import os
import time
from tensorboardX import SummaryWriter
from config import PARAMS_CONFIG
from man_utils.utils import (
    get_params,
    get_optimizer,
    get_scheduler,
    save_checkpoint,
    load_checkpoint,
    save_params,
)
from dataset import get_train_data, batchify
import model
from trainer import lm_train_iter, dis_train_iter


def main(walk_params, model_params, optim_params, data_params, trainer_params):
    hyper_param = trainer_params['hyper_param'], 
    hp_value = trainer_params[trainer_params['hyper_param']]
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print('walk_params:', walk_params, '\nmodel_parmas:', model_params,
          '\noptim_params:', optim_params, '\ndata_params:', data_params,
          '\ntrainer_params:', trainer_params)

    assert torch.cuda.is_available()
    device = torch.device('cuda')

    torch.manual_seed(trainer_params['seed'])
    torch.cuda.manual_seed_all(trainer_params['seed'])

    # data
    working_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(working_dir)

    data_dir = os.path.join(working_dir, 'data')
    corpus_dir = os.path.join(data_dir,
                              'train/{}'.format(data_params['dataset']))

    assert os.path.exists(corpus_dir)
    data_params['corpus_dir'] = corpus_dir

    corpus = get_train_data(data_params, walk_params['window_size'],
                            walk_params['p'], walk_params['q'],
                            walk_params['walk_mode'],
                            trainer_params['batch_size'], device)
    train_data = corpus.train
    vocab = torch.LongTensor(list(corpus.vocab.values())).to(device)
    # adj = corpus.adj.to(device)
    # attr = corpus.attr.to(device)
    attr = corpus.attr
    # attrs_sim = corpus.attrs_sim.to(device)
    attrs_sim = corpus.attrs_sim
    as_indices = torch.argsort(attrs_sim, descending=True)
    k = trainer_params['k']
    assert 0 < k < corpus.vocab_size // 2
    # attrs_sim[torch.range(0, corpus.vocab_size - 1).long().unsqueeze(1),
    #           as_indices[:, k: -k]] = 0
    model_params['attr_size'] = attr.size(1)

    # model
    embedding_model = model.EmbeddingModel(
        'embedding_model',
        corpus.vocab_size,
        info_mode=trainer_params['info_mode'],
        **model_params).to(device)

    if trainer_params['is_attr'] == 0:
        emb_dim = model_params['embedding_dim']
    elif trainer_params['info_mode'] == 'cat':
        emb_dim = model_params['embedding_dim'] * 2
    elif trainer_params['info_mode'] == 'add':
        emb_dim = model_params['embedding_dim']
    lm_model = model.Decoder('TransformerDecoder',
                             vocab_size=corpus.vocab_size,
                             layer_num=model_params['layer_num'],
                             d_model=emb_dim,
                             h=model_params['heads_num'],
                             d_ff=model_params['hidden_dim'],
                             dropout=model_params['dropout']).to(device)
    # optimizer & scheduler
    lm_optim = get_optimizer([embedding_model, lm_model],
                             optim=optim_params['lm_optim'],
                             lr=optim_params['lm_lr'],
                             momentum=optim_params['momentum'],
                             weight_decay=optim_params['weight_decay'])
    lm_scheduler_decay = get_scheduler(lm_optim, lr_mode='decay')
    lm_scheduler_warmup = get_scheduler(lm_optim,
                                        lr_mode='raise',
                                        lr_warmup=optim_params['lm_lr_warmup'])

    dis_optim = get_optimizer([embedding_model],
                              optim=optim_params['dis_optim'],
                              lr=optim_params['dis_lr'],
                              momentum=optim_params['momentum'],
                              weight_decay=optim_params['weight_decay'])
    dis_scheduler = get_scheduler(dis_optim, lr_mode='decay')
    writer = SummaryWriter(comment='%s_%s_%s' % (data_params['dataset'], hyper_param, hp_value))

    # checkpoint
    checkpoint_path = os.path.join(
        working_dir, 'checkpoint/{}/{}_{}.pt'.format(
            data_params['dataset'], hyper_param, hp_value))
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path))
    iter_init = load_checkpoint(
        checkpoint_path, [embedding_model, lm_model], [lm_optim, dis_optim],
        [lm_scheduler_warmup, lm_scheduler_decay, dis_scheduler])

    # train
    nb_batches_per_iter = trainer_params['batch_per_iter']
    # nb_batches_per_lm_iter = len(train_data) // trainer_params['batch_size']
    nb_batches_per_lm_iter = nb_batches_per_iter
    if trainer_params['dis_ramdom'] == 0:
        nb_batches_per_dis_iter = corpus.vocab_size // trainer_params[
            'block_size']
    else:
        nb_batches_per_dis_iter = nb_batches_per_iter
    nb_lm_iter = trainer_params['lm_iters']
    nb_dis_iter = trainer_params['dis_iters']

    train_pos = (0 if iter_init == 0 else
                 (iter_init - 1)) * nb_batches_per_lm_iter % len(
                     train_data)  
    start_time = time.time()
    dis_loss = None
    lm_loss = None
    lm_acc = None
    is_attr = trainer_params['is_attr']
    train_data = batchify(train_data, trainer_params['batch_size'])
    for iter_no in range(iter_init, trainer_params['iters']):
        print(f'iters {iter_no + 1}/{trainer_params["iters"]}')
        for i in range(nb_lm_iter):
            print(f'epoch: {iter_no + 1}, lm_iter: {i + 1}/{nb_lm_iter}')
            lm_steps = (iter_no * nb_lm_iter + i) * nb_batches_per_lm_iter
            if lm_steps < optim_params['lm_lr_warmup']:
                lm_scheduler = lm_scheduler_warmup
            else:
                lm_scheduler = lm_scheduler_decay
            lm_loss, lm_acc, train_pos = lm_train_iter(
                lm_steps,
                writer=writer,
                lm_model=lm_model,
                emb_model=embedding_model,
                optimizer=lm_optim,
                scheduler=lm_scheduler,
                batch_size=trainer_params['batch_size'],
                # nb_batches_per_iter=nb_batches_per_iter,
                nb_batches_per_iter=nb_batches_per_lm_iter,
                data=train_data,
                attr=attr if trainer_params['is_attr'] == 1 else None,
                train_pos=train_pos,
                grad_clip=optim_params['grad_clip'],
                warmup_steps=optim_params['lm_lr_warmup'])

        for j in range(nb_dis_iter):
            print('epoch: {}, dis_iter: {}/{}'.format(iter_no, j, nb_dis_iter))
            dis_steps = (iter_no * nb_dis_iter + j) * nb_batches_per_dis_iter
            dis_loss = dis_train_iter(
                dis_steps,
                writer=writer,
                emb_model=embedding_model,
                optimizer=dis_optim,
                scheduler=dis_scheduler,
                adj=attrs_sim,
                data=vocab,
                beta=trainer_params['beta'],
                attr=attr if trainer_params['is_attr'] == 1 else None,
                nb_batches_per_iter_max=nb_batches_per_dis_iter,
                block_size=trainer_params['block_size'],
                device=device,
                mode=trainer_params['dis_loss_mode'],
                indices=torch.cat([as_indices[:, :k], as_indices[:, -k:]],
                                  dim=-1))

        save_checkpoint(
            checkpoint_path, iter_no, [embedding_model, lm_model],
            [lm_optim, dis_optim],
            [lm_scheduler_warmup, lm_scheduler_decay, dis_scheduler])
    end_time = time.time()
    print('{} epochs cost {:.1f} hours.'.format(
        trainer_params['iters'], (end_time - start_time) / 3600))
    writer.close()

    output_dir = os.path.join(working_dir,
                              'features/{}'.format(data_params['dataset']))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # embeddings = lm_model.generator.output.weight.cpu().detach().numpy()
    embeddings = embedding_model(vocab, attr=attr.cuda(), info_mode=trainer_params['info_mode'])
    embeddings = embeddings.detach().cpu()
    np.save(
                os.path.join(
                    output_dir,
                    '{}_{}_{}_{:.2f}_{:.2f}.npy'.format(
                        time.strftime('%m%d_%H.%M', time.localtime()),
                        hyper_param, hp_value,
                        lm_acc.item() if lm_acc is not None else -1,
                        dis_loss if dis_loss is not None else -1)), embeddings)
   
    log_dir = os.path.join(working_dir, 'logs/%s' % data_params['dataset'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_params(log_dir,
                lm_loss=lm_loss,
                lm_acc=lm_acc,
                walk_params=walk_params,
                model_params=model_params,
                optim_params=optim_params,
                data_params=data_params,
                trainer_params=trainer_params)


if __name__ == '__main__':
    main(**get_params(params_config=PARAMS_CONFIG))
