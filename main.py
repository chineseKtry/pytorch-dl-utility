from __future__ import print_function
import time
import numpy as np
import sys
import h5py
import cPickle
import argparse
import subprocess
import shutil
import os
from tempfile import mkdtemp
from sklearn.metrics import accuracy_score, roc_auc_score
from pprint import pprint

import torch
import ray
from ray import tune

from hyperband import Hyperband
import util


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch + Hyperband for genomics')
    parser.add_argument('--hyper', dest='hyper', default=False,
                        action='store_true', help='Perform hyper-parameter tuning')
    parser.add_argument('--debug', dest='debug', default=False,
                        action='store_true', help='Debug mode, run one iteration')

    parser.add_argument('--out-dir', dest='out_dir', help='Result directory')
    parser.add_argument('--data', dest='data', help='Data path or directory')
    parser.add_argument('--model', dest='model', help='Path to python file with PyTorch model')

    parser.add_argument('--train_epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--batch_size', default=100, type=int,
                        help='Batch size in gradient-based training')

    # parser.add_argument('-t', '--train', dest='train', default=False, action='store_true',
    #                     help='Train on the training set with the best hyper-params')
    # parser.add_argument('-e', '--eval', dest='eval', default=False,
    #                     action='store_true', help='Evaluate the model on the test set')
    # parser.add_argument('-p', '--predict', dest='infile', default='',
    #                     help='Path to data to predict on (up till batch number)')
    # parser.add_argument('-m', '--model', dest='model',
    #                     help='Path to the model file')
    # parser.add_argument('-o', '--outdir', dest='outdir', default='',
    #                     help='Output directory for the prediction on new data')
    # parser.add_argument('-hi', '--hyper_epochs', dest='hyper_epochs', default=20,
    #                     type=int, help='Num of max epochs for each hyper-param config')
    # parser.add_argument('-te', '--train_epochs', default=20,
    #                     type=int, help='The number of epochs to train for')
    # parser.add_argument('-pa', '--patience', default=10, type=int,
    #                     help='number of epochs with no improvement after which training will be stopped.')
    # parser.add_argument('-bs', '--batch_size', default=100,
    #                     type=int, help='Batch_size in SGD-based training')
    # parser.add_argument('-w', '--state_file', default=None,
    #                     help='Weight file for the best model')
    # parser.add_argument('-l', '--last_state_file', default=None,
    #                     help='Weight file after training')
    # parser.add_argument('-r', '--retrain', default=None,
    #                     help='codename for the retrain run')
    # parser.add_argument('-rw', '--retrain_state_file', default='',
    #                     help='Weight file to load for retraining')
    # parser.add_argument('-dm', '--datamode', default='memory',
    #                     help='whether to load data into memory ("memory") or using a generator("generator")')
    # parser.add_argument('-ei', '--evalidx', dest='evalidx', default=0, type=int,
    #                     help='which output neuron (0-based) to calculate 2-class auROC for')
    # parser.add_argument('--epoch_ratio', default=1, type=float,
    #                     help='when training with data generator, optionally shrink each epoch size by this factor to enable more frequen evaluation on the valid set')
    # parser.add_argument('-shuf', default=1, type=int,
    #                     help='whether to shuffle the data at the begining of each epoch (1/0)')

    return parser.parse_args()


def run_train(model, weightfile2save):
    checkpointer = ModelCheckpoint(
        filepath=weightfile2save, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=args.patience, verbose=0)
    if args.datamode == 'generator':
        trainbatch_num, train_size = hb.probedata(
            join(args.topdir, 'train.h5.batch'))
        validbatch_num, valid_size = hb.probedata(
            join(args.topdir, 'valid.h5.batch'))
        history_callback = model.fit_generator(
            hb.BatchGenerator(args.batch_size, join(
                args.topdir, 'train.h5.batch'), shuf=args.shuf == 1),
            train_size / args.batch_size * args.epoch_ratio,
            args.train_epochs,
            validation_data=hb.BatchGenerator(args.batch_size, join(
                args.topdir, 'valid.h5.batch'), shuf=args.shuf == 1),
            validation_steps=np.ceil(float(valid_size) / args.batch_size),
            callbacks=[checkpointer, early_stopping])
    else:
        Y_train, X_train = hb.readdata(join(args.topdir, 'train.h5.batch'))
        Y_val, X_val = hb.readdata(join(args.topdir, 'valid.h5.batch'))
        model.train(X_train, Y_train, args.train_epochs, args.batch_size)
        util.train_pytorch(model, X_train, Y_train)  # TODO checkpoint and early stopping
        history_callback = model.fit(
            X_train,
            Y_train,
            batch_size=args.batch_size,
            epochs=args.train_epochs,
            validation_data=(X_val, Y_val),
            callbacks=[checkpointer, early_stopping],
            shuffle=args.shuf == 1)
    return model, history_callback

if __name__ == '__main__':
    args = parse_args()
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    args.data = os.path.abspath(args.data)

    model_dir = os.path.join(args.out_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if os.path.exists('model_def.py'):
        os.remove('model_def.py')
    os.symlink(args.model, 'model_def.py')
    import model_def

    ray.init()  # TODO
    sched = ray.tune.schedulers.HyperBandScheduler(time_attr='training_iteration', reward_attr='reward')

    model_def.config_generator.update(vars(args))
    experiment_config = {
        'stop': {
            'training_iteration': args.train_epochs
        },
        'trial_resources': {
            'cpu': 4  # TODO
        },
        'run': model_def.Model,
        # TODO checkpoint_freq
        'checkpoint_at_end': True,
        'config': model_def.config_generator,
        'num_samples': 15 # TODO
    }
    if args.debug:
        experiment_config['stop']['training_iteration'] = 1
        experiment_config['trial_resources']['cpu'] = 1
        experiment_config['num_samples'] = 1
        experiment_config['max_failures'] = 1

    tune.run_experiments({
        model_name: experiment_config,
    }, verbose=0, scheduler=sched)

if __name__ == '__main_':
    args = parse_args()
    model_name = splitext(basename(args.model))[0]

    outdir = join(args.topdir, model_name)
    if not exists(outdir):
        os.makedirs(outdir)

    param_file = join(outdir, 'best_hyperparameters.json')
    last_state_base = join(outdir, 'last_state')
    # state_file = join(outdir,model_arch+'_bestmodel_weights.h5') if args.state_file is None else args.state_file
    # evalout = join(outdir,model_arch+'_eval.txt')

    shutil.copy(args.model, join(outdir, 'model_def.py'))
    sys.path.append(outdir)
    import model_def

    hb = Hyperband(model_def.get_params, model_def.try_params,
                   args.topdir, max_epochs=args.hyper_epochs, datamode=args.datamode)

    if args.hyper:
        # Hyper-parameter tuning
        results = hb.run(skip_last=1)

        best_result = sorted(results, key=lambda x: x['loss'])[0]
        pprint(best_result['params'])

        util.save_json(best_result['params'], param_file)

    if args.train:
        # Training
        model = model_def.get_model(util.load_json(param_file))

        Y_train, X_train = hb.readdata(join(args.topdir, 'train.h5.batch'))
        Y_val, X_val = hb.readdata(join(args.topdir, 'valid.h5.batch'))
        model.train(X_train, Y_train, args.train_epochs, args.batch_size)

        model.save_checkpoint(last_state_base)
        # all_hist = np.asarray([myhist['loss'], myhist['categorical_accuracy'], myhist[
        #                       'val_loss'], myhist['val_categorical_accuracy']]).transpose()
        # np.savetxt(join(outdir, model_arch + '.training_history.txt'),
        #            all_hist, delimiter='\t', header='loss\tacc\tval_loss\tval_acc')

    if args.retrain:
        # Resume training
        new_weight_file = state_file + '.' + args.retrain
        new_last_weight_file = last_weight_file + '.' + args.retrain

        model = model_def.get_model(util.load_json(
            param_file)).load_checkpoint(args.retrain_state_file)
        model, history_callback = run_train(model, new_weight_file)

        model.save_weights(new_last_weight_file, overwrite=True)
        system('touch ' + join(outdir, model_arch + '.traindone'))
        myhist = history_callback.history
        all_hist = np.asarray([myhist['loss'], myhist['categorical_accuracy'], myhist[
                              'val_loss'], myhist['val_categorical_accuracy']]).transpose()
        np.savetxt(join(outdir, model_arch + '.training_history.' + args.retrain +
                        '.txt'), all_hist, delimiter='\t', header='loss\tacc\tval_loss\tval_acc')

    if args.eval:
        # Evaluate
        model = model_def.get_model(util.load_json(
            param_file)).load_checkpoint(last_state_base + '.pth')

        pred_for_evalidx = []
        pred_bin = []
        y_true_for_evalidx = []
        y_true = []
        testbatch_num, _ = hb.probedata(join(args.topdir, 'test.h5.batch'))
        test_generator = hb.BatchGenerator(None, join(
            args.topdir, 'test.h5.batch'), shuf=args.shuf == 1)
        for _ in range(testbatch_num):
            X_test, Y_test = test_generator.next()
            t_pred = model.predict(X_test).detach().numpy()
            pred_for_evalidx += [x[args.evalidx] for x in t_pred]
            pred_bin += [np.argmax(x) for x in t_pred]
            y_true += [np.argmax(x) for x in Y_test]
            y_true_for_evalidx += [x[args.evalidx] for x in Y_test]

        t_auc = roc_auc_score(y_true_for_evalidx, pred_for_evalidx)
        t_acc = accuracy_score(y_true, pred_bin)
        print('Test AUC for output neuron {}:'.format(args.evalidx), t_auc)
        print('Test categorical accuracy:', t_acc)
        np.savetxt(evalout, [t_auc, t_acc])

    if args.infile != '':
        # Predict on new data
        model = model_def.get_model(util.load_json(param_file)).load_checkpoint(state_file)

        predict_batch_num, _ = hb.probedata(args.infile)
        print('Total number of batch to predict:', predict_batch_num)

        outdir = join(dirname(args.infile), '.'.join(
            ['pred', model_arch, basename(args.infile)])) if args.outdir == '' else args.outdir
        if exists(outdir):
            print('Output directory', outdir, 'exists! Overwrite? (yes/no)')
            if raw_input().lower() == 'yes':
                system('rm -r ' + outdir)
            else:
                print('Quit predicting!')
                sys.exit(1)

        for i in range(predict_batch_num):
            print('predict on batch', i)
            batch_data = h5py.File(args.infile + str(i + 1), 'r')['data']

            time1 = time.time()
            pred = model.predict(batch_data)
            time2 = time.time()
            print('predict took %0.3f ms' % ((time2 - time1) * 1000.0))

            t_outdir = join(outdir, 'batch' + str(i + 1))
            os.makedirs(t_outdir)
            for label_dim in range(pred.shape[1]):
                with open(join(t_outdir, str(label_dim) + '.pkl'), 'wb') as f:
                    cPickle.dump(pred[:, label_dim], f)
