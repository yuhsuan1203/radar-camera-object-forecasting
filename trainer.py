import numpy as np
import metrics
import torch
import torch.nn as nn
from tqdm import tqdm


def train_seqseq(encoder, decoder, device, train_loader, encoder_optimizer, decoder_optimizer, epoch, loss_function, learning_rate):
    encoder.train()
    decoder.train()
    total_loss = 0
    ades = []
    fdes = []
    for batch_idx, data in enumerate(tqdm(train_loader)):

        features, labels = data['features'].to(
            device), data['labels'].to(device) #, data['dtp_features'].to(device)

        features = features.float()
        labels = labels.float()
        #dtp_features = dtp_features.float()

        context = encoder(features)

        output = decoder(context, val=False)

        #print('output\n', list(output))
        #print('labels\n', list(labels))
        loss = loss_function(output, labels)
        #print('loss\n', loss)

        ades.append(list(metrics.calc_ade(output.cpu().detach().numpy(), 
                                          labels.cpu().detach().numpy(), return_mean=False)))
        fdes.append(list(metrics.calc_fde(output.cpu().detach().numpy(),
                                          labels.cpu().detach().numpy(), 24, return_mean=False)))

        # Backward and optimize
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        # Clip gradients
        nn.utils.clip_grad_norm(decoder.parameters(), 1)
        for p in decoder.parameters():
            p.data.add_(-learning_rate, p.grad.data)
        total_loss += loss
    # Flatten lists
    ades = [item for sublist in ades for item in sublist]
    fdes = [item for sublist in fdes for item in sublist]

    print('Train ADE: ', np.round(np.mean(ades), 1))
    print('Train FDE: ', np.round(np.mean(fdes), 1))
    print('Train loss: ', total_loss.cpu().detach().numpy())


def test_seqseq(encoder, decoder, device, test_loader, loss_function, return_predictions=False, phase='Val'):
    encoder.eval()
    decoder.eval()
    ades = []
    fdes = []

    ades2 = []
    fdes2 = []

    outputs = np.array([])
    targets = np.array([])
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            # if batch_idx == 5:
            #     break
            features, labels = data['features'].to(device), data['labels'].to(
                device)  #, data['dtp_features'].to(device)
            features = features.float()
            labels = labels.float()
            #dtp_features = dtp_features.float()
            context = encoder(features, val=True)
            output = decoder(context, val=True)

            allzeros_nparray = np.zeros(labels.cpu().numpy().shape)
            
            ades.append(list(metrics.calc_ade(output.cpu().numpy(),
                                              labels.cpu().numpy(), return_mean=False)))
            fdes.append(list(metrics.calc_fde(output.cpu().numpy(),
                                              labels.cpu().numpy(), 24, return_mean=False)))
            
            
            ades2.append(list(metrics.calc_ade(allzeros_nparray,
                                               labels.cpu().numpy(), return_mean=False)))
            fdes2.append(list(metrics.calc_fde(allzeros_nparray,
                                               labels.cpu().numpy(), 24, return_mean=False)))
            
            if return_predictions:
                outputs = np.append(outputs, output.cpu().numpy())
                targets = np.append(targets, labels.cpu().numpy())

    # Flatten lists
    ades = [item for sublist in ades for item in sublist]
    fdes = [item for sublist in fdes for item in sublist]

    print(phase + ' ADE: ' + str(np.round(np.mean(ades), 1)))
    print(phase + ' FDE: ' + str(np.round(np.mean(fdes), 1)))

    ades2 = [item for sublist in ades2 for item in sublist]
    fdes2 = [item for sublist in fdes2 for item in sublist]

    #print(len(ades2))
    #print(len(fdes2))
    #print(ades2)
    #print(fdes2)
    print('CVCS ' + phase + ' ADE: ' + str(np.round(np.mean(ades2), 1)))
    print('CVCS ' + phase + ' FDE: ' + str(np.round(np.mean(fdes2), 1)))


    return outputs, targets, np.mean(ades), np.mean(fdes)
