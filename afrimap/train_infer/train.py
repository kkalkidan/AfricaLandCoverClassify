import sys
import pandas as pd
import torch
from afrimap.train_infer.metric import confusion_matrix_local, accuracy, confident_only_accuracy, confident_only_cm, f1_score_local, confident_f1_score
import torch.nn as nn
from afrimap.train_infer.utils import create_configs, get_train_val


def evaluate(model, dataset_loader, device, criterion):
    model.eval()
    # criterion = cross_entropy(class_weights)
    # lists that accumulate mean losses and accuracies for each batch
    losses = []
    confident_accuracies = []
    overall_accuracies = []
    # confusion matrix accumulators for each batch
    cm = None 
    confident_cm = None

    f1_scores = []
    conf_f1_scores = []

    # df = pd.DataFrame([])
    with torch.no_grad():
         # run each batch in the train_loader 
        for data, target, unmasked_target, score, _ in dataset_loader:

            data, target, unmasked_target, score = data.to(device), \
                                                    target.to(device), \
                                                    unmasked_target.to(device), \
                                                    score.to(device)
            # predict input 
            output = model(data)
            # calculate loss
            loss = criterion(output, target)
            # overall accuracy and confident accuracy for the batch 
            overall_accuracy = accuracy(output, unmasked_target)
            confident_accuracy = confident_only_accuracy(output, unmasked_target, score)

            # setting confusion matrix 
            if cm is None: cm = confusion_matrix_local(output, unmasked_target)
            else:  cm += confusion_matrix_local(output, unmasked_target)
            if confident_cm is None: confident_cm = confident_only_cm(output, unmasked_target, score)
            else: confident_cm += confident_only_cm(output, unmasked_target, score)

            # appending the batch loss to the loss list and same for others 
            losses.append(loss)
            overall_accuracies.append(overall_accuracy)
            confident_accuracies.append(confident_accuracy)
            f1_scores.append(f1_score_local(output, unmasked_target))
            conf_f1_scores.append(confident_f1_score(output, unmasked_target, score))
    # calculate the mean loss for all training dataset(for a single epoch) and append it to the lr loss
    # same thing for the other variables 
    val_loss = torch.tensor(losses).mean().item()
    val_accus = torch.tensor(overall_accuracies).nan_to_num().mean().item()
    val_conf_accus = torch.tensor(confident_accuracies).nan_to_num().mean().item()
    val_f1_score = torch.tensor(f1_scores).nan_to_num().mean().item()
    val_conf_f1 = torch.tensor(conf_f1_scores).nan_to_num().mean().item()

    model.train()
    return val_loss, val_accus, val_conf_accus, val_f1_score, val_conf_f1


def train(image_path, label_path, lr, nb_epochs, tr):
 
    ktrain_loader, kval_loader,weights = get_train_val(image_path, label_path)
    for k in ktrain_loader:
        print(k[0].shape)
        break
    early_stop_count = 0
    df_lr = pd.DataFrame([])
    # specify the type arrangement 
    # df_lr['type'] = ['loss']*nb_epochs + ['overall_accu']*nb_epochs + ['conf_accus']*nb_epochs
    prev_val_accu = 0
    max_val_accu = 0
    config_params = create_configs(lr=lr, class_weights=weights)
    model, device, criterion, optimizer, lr_scheduler = config_params
    # lists that accumulate the mean loss and accuracies for each lr in 10 epochs 
    lr_losses = []
    lr_accus = []
    lr_conf_accus = []
    lr_f1s = []
    lr_conf_f1s = []


    lr_val_losses = []
    lr_val_accus = []
    lr_val_conf_accus = []
    lr_val_f1s = []
    lr_val_conf_f1s = []
    try:
        # run the epochs 
        for epoch in range(1, nb_epochs+1):
            # lists that accumulate mean losses and accuracies for each batch
            losses = []
            confident_accuracies = []
            overall_accuracies = []
            f1_scores = []
            conf_f1_scores = []
            # confusion matrix accumulators for each batch
            cm = None 
            confident_cm = None
            # run each batch in the train_loader 
            for data, target, unmasked_target, score, _ in ktrain_loader:

                data, target, unmasked_target, score = data.to(device), \
                                                        target.to(device), \
                                                        unmasked_target.to(device), \
                                                        score.to(device)
                # zero optimizer grads
                optimizer.zero_grad()
                # predict input 
                output = model(data)
                # calculate loss
                loss = criterion(output, target)
                # find gradients 
                loss.backward()
                # step in the direction of the gradients 
                optimizer.step()
                
                # overall accuracy and confident accuracy for the batch 
                overall_accuracy = accuracy(output, unmasked_target)
                confident_accuracy = confident_only_accuracy(output, unmasked_target, score)

                # setting confusion matrix 
                if cm is None: cm = confusion_matrix_local(output, unmasked_target)
                else:  cm += confusion_matrix_local(output, unmasked_target)
                if confident_cm is None: confident_cm = confident_only_cm(output, unmasked_target, score)
                else: confident_cm += confident_only_cm(output, unmasked_target, score)

                # appending the batch loss to the loss list and same for others 
                losses.append(loss.item())
                overall_accuracies.append(overall_accuracy)
                confident_accuracies.append(confident_accuracy)
                f1_scores.append(f1_score_local(output, unmasked_target))
                conf_f1_scores.append(confident_f1_score(output, unmasked_target, score))
            # print optimizers current learning rate for sanity check=
            print('lr from optimizer', optimizer.state_dict()['param_groups'][0]['lr'])
            # update lr_rate
            lr_scheduler.step()
                # print('lr from the scheduler(print_lr)', lr_scheduler.print_lr())
            print('lr from the scheduler(get_last_lr', lr_scheduler.get_last_lr())
            # calculate the mean loss for all training dataset(for a single epoch) and append it to the lr loss
            # same thing for the other variables 
            lr_losses.append(torch.tensor(losses).mean().item())
            lr_accus.append(torch.tensor(overall_accuracies).nan_to_num().mean().item())
            lr_conf_accus.append(torch.tensor(confident_accuracies).nan_to_num().mean().item())
            lr_f1s.append(torch.tensor(f1_scores).nan_to_num().mean().item())
            lr_conf_f1s.append(torch.tensor(conf_f1_scores).nan_to_num().mean().item())

            
            print(f"Epoch {epoch}, train loss: {torch.tensor(losses).mean().item():.4f} \t train acc: \
            {torch.tensor(overall_accuracies).nan_to_num().mean().item():5f} \t  confident train acc: \
            {torch.tensor(confident_accuracies).nan_to_num().mean().item():5f}")
            # at the last epoch show the confusion matrix 
            if (epoch == nb_epochs):
                torch.save(model.state_dict(), f'afrimap/train_infer/output/manet_{lr}_{epoch}.pth')

            val_loss, val_accus, val_conf_accus, val_f1, val_conf_v1 = evaluate(model, kval_loader, device, criterion)
            lr_val_losses.append(val_loss)
            lr_val_accus.append(val_accus)
            lr_val_conf_accus.append(val_conf_accus) 
            lr_val_f1s.append(val_f1)
            lr_val_conf_f1s.append(val_conf_v1)
            print(f"\t val loss: {val_loss:.4f} \t val acc: {val_accus :5f} \t confident val acc: {val_conf_accus:5f}") 
            if(val_accus <= prev_val_accu):
                early_stop_count += 1
            else:
                prev_val_accu = val_accus
                early_stop_count = 0
            if(val_accus > max_val_accu):
                max_val_accu = val_accus
                print('###### saving model ######')
                torch.save(model.state_dict(), f'afrimap/train_infer/output/manet_best.pth')

    except Exception as e:
        print(e)
    finally:
        print('writing df and closing pdf')
        df_lr['epoch'] = list(range(1, nb_epochs+1))
        df_lr.loc[range(len(lr_losses)), 'loss'] = lr_losses
        df_lr.loc[range(len(lr_accus)), 'accu'] = lr_accus
        df_lr.loc[range(len(lr_conf_accus)), 'conf_accu'] = lr_conf_accus
        df_lr.loc[range(len(lr_val_losses)), 'val_loss'] = lr_val_losses
        df_lr.loc[range(len(lr_val_accus)), 'val_accu'] = lr_val_accus
        df_lr.loc[range(len(lr_val_conf_accus)), 'val_conf_accu'] = lr_val_conf_accus
        df_lr.loc[range(len(lr_f1s)), 'f1'] = lr_f1s
        df_lr.loc[range(len(lr_conf_f1s)), 'conf_f1'] = lr_conf_f1s
        df_lr.loc[range(len(lr_val_f1s)), 'val_f1'] = lr_val_f1s
        df_lr.loc[range(len(lr_val_conf_f1s)), 'val_conf_f1'] = lr_val_conf_f1s
        df_lr.to_csv(f'afrimap/train_infer/output/manet_{nb_epochs}.csv', index = False) 

