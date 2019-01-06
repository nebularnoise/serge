import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data

###############################################################################
#### MISC.

def save_def(seed,mname,mpath,instruments,style,n_ep,batch,lr,lr_decay,lr_step,optim,lambda_SC,\
             lambda_logmag,sr,n_fft,win_size,hop_size,n_MCNN,n_sig_out,n_mels,fmin,fmax,f_bins,\
             bn,n_heads,n_conv,chan_down,stride_up,width_up,padding_in,eps,grad_clip_thresh,weight_decay,log_scaling,mag_thr):
    model_def = {}
    model_def.update({'seed':seed})
    model_def.update({'instruments':instruments})
    model_def.update({'style':style})
    model_def.update({'n_ep':n_ep})
    model_def.update({'batch':batch})
    model_def.update({'lr':lr})
    model_def.update({'lr_decay':lr_decay})
    model_def.update({'lr_step':lr_step})
    model_def.update({'optim':optim})
    model_def.update({'lambda_SC':lambda_SC})
    model_def.update({'lambda_logmag':lambda_logmag})
    model_def.update({'sr':sr})
    model_def.update({'n_fft':n_fft})
    model_def.update({'win_size':win_size})
    model_def.update({'hop_size':hop_size})
    model_def.update({'n_MCNN':n_MCNN})
    model_def.update({'n_sig_out':n_sig_out})
    model_def.update({'n_mels':n_mels})
    model_def.update({'fmin':fmin})
    model_def.update({'fmax':fmax})
    model_def.update({'f_bins':f_bins})
    model_def.update({'bn':bn})
    model_def.update({'n_heads':n_heads})
    model_def.update({'n_conv':n_conv})
    model_def.update({'chan_down':chan_down})
    model_def.update({'stride_up':stride_up})
    model_def.update({'width_up':width_up})
    model_def.update({'padding_in':padding_in})
    model_def.update({'eps':eps})
    model_def.update({'grad_clip_thresh':grad_clip_thresh})
    model_def.update({'weight_decay':weight_decay})
    model_def.update({'log_scaling':log_scaling})
    model_def.update({'mag_thr':mag_thr})
    np.save(mpath+mname+'_definition.npy',model_def)
    return model_def

def build_pretrainedMCNN(mname,mpath,build_flag,device,eval_mod,n_ep=None):

    model_def = np.load(mpath+mname+'_definition.npy').item()

    instruments = model_def['instruments']
    style = model_def['style']
    if n_ep is None:
        n_ep = model_def['n_ep']
        # final training epoch
        print('importing final training epoch = '+str(n_ep))
    else:
        print('importing intermediate training epoch = '+str(n_ep))

    n_conv = model_def['n_conv']
    chan_down = model_def['chan_down']
    width_up = model_def['width_up']
    padding_in = model_def['padding_in']
    n_heads = model_def['n_heads']
    n_MCNN = model_def['n_MCNN']
    n_sig_out = model_def['n_sig_out']
    stride_up = model_def['stride_up']
    bn = model_def['bn']
    if 'log_scaling' in model_def:
        log_scaling = model_def['log_scaling']
    else:
        log_scaling = 0
    if 'mag_thr' in model_def:
        mag_thr = model_def['mag_thr']
    else:
        mag_thr = None

    if build_flag==1:
        MCNN = MCNN_net(n_conv,chan_down,width_up,padding_in,n_heads,n_MCNN,n_sig_out,stride_up,bn)
        MCNN.to(device)
        if str(device)!='cpu' and torch.cuda.is_available():
            states = torch.load(mpath+mname+'_'+str(n_ep)+'.pth', map_location=lambda storage, loc: storage.cuda(device))
        else:
            states = torch.load(mpath+mname+'_'+str(n_ep)+'.pth', map_location='cpu')
        MCNN.load_state_dict(states['state'])
        if eval_mod==0:
            MCNN.train()
        if eval_mod==1:
            MCNN.eval()
    else:
        MCNN = None

    return MCNN,model_def,instruments,style,n_ep,log_scaling,mag_thr


###############################################################################
#### LOSSES

def mag_loss(input_amp,output_amp,lambda_SC,lambda_logmag,eps):
    # xx_amp is magnitude (possibly mapped on mel scale)
    # the losses are scaled by input magnitude

    ### Compute log magnitude differences
    log_stft_mag_diff = torch.sum(torch.abs(torch.log(input_amp + eps) - torch.log(output_amp + eps)), (1,2))
    log_stft_ref = torch.sum(torch.abs(torch.log(input_amp + eps)), (1,2))
    log_stft_mag = torch.mean(log_stft_mag_diff/(log_stft_ref + eps))

    total_loss = lambda_logmag * log_stft_mag

    if lambda_SC!=0:
        ### Compute spectral convergence
        frobenius_diff = torch.sqrt(torch.sum(torch.pow(input_amp - output_amp,2),(1,2)))

        frobenius_input_amp = torch.sqrt(torch.sum(torch.pow(input_amp,2),(1,2)))
        spectral_convergence = torch.mean(frobenius_diff/(frobenius_input_amp))

        # it goes inf or very high --> normalize by dimensionnality instead of input amp ...
        #spectral_convergence = torch.mean(frobenius_diff/(input_amp.shape[1]*input_amp.shape[2]))
        total_loss += lambda_SC * spectral_convergence
    else:
        spectral_convergence = torch.tensor([0.])

    return total_loss,(lambda_SC*spectral_convergence).detach().item(),(lambda_logmag*log_stft_mag).detach().item()


###############################################################################
#### MCNN ARCHITECTURE

class tr_head(nn.Module):
    def __init__(self,chan_down,width_up,padding_in,n_conv,stride_up,bn):
        super(tr_head, self).__init__()
        self.deconvolutions = nn.ModuleList()
        for conv_id in range(n_conv):
            deconv = nn.ConvTranspose1d(chan_down[conv_id],chan_down[conv_id+1],\
                                width_up[conv_id],stride=stride_up,padding=padding_in[conv_id],\
                                output_padding=0,bias=True)
            nn.init.xavier_uniform_(deconv.weight.data)
            self.deconvolutions.append(deconv)
            if bn==1:
                self.deconvolutions.append(nn.BatchNorm1d(chan_down[conv_id+1]))
            self.deconvolutions.append(nn.ELU())
        self.scale_out = nn.Parameter(torch.tensor([1.]))
    def forward(self,x):
        # spectro_slices should have the shape (batch,f_bins,n_MCNN)
        for i, deconv in enumerate(self.deconvolutions):
            x = deconv(x)
        x = x.squeeze(1).mul(self.scale_out)
        return x

class MCNN_net(nn.Module):
    def __init__(self,n_conv,chan_down,width_up,padding_in,n_heads,n_MCNN,n_sig_out,stride_up,bn):
        super(MCNN_net, self).__init__()
        print('building a basic MCNN network with '+str(n_heads)+' heads')
        if bn==1:
            print('using 1D batch-norm')
        self.n_heads = n_heads
        self.heads = nn.ModuleList()
        for head_id in range(n_heads):
            print('head '+str(head_id))
            self.heads.append(tr_head(chan_down,width_up,padding_in,n_conv,stride_up,bn))
        self.a_softsign = nn.Parameter(torch.tensor([1.]))
        self.b_softsign = nn.Parameter(torch.tensor([2.]))
    def forward(self,mb_mag):
        # mb_mag should have the shape (batch,f_bins,n_MCNN)
        signal_slices = sum([head(mb_mag) for head in self.heads])
        mb_sigout = (signal_slices*self.a_softsign)/(1+torch.abs(signal_slices*self.b_softsign))
        return mb_sigout


###############################################################################
#### DATA UTILS

def datasets_tr_ev_test(data_path,instruments,inst_ref,n_sig_out,style):

    print('importing data from '+data_path)

    train_sig_slices = []
    train_labels = []
    eval_sig_slices = []
    eval_labels = []
    test_sig_slices = []
    test_labels = []

    for inst in instruments:
        print('importing '+inst_ref[inst][0])

        train_slices = 0
        eval_slices = 0
        test_slices = 0

        train_dic = np.load(data_path+inst_ref[inst][0]+'_traindic.npy').item()
        # should contain {file_id:[np(waveform),[labels],'file_name']}
        # labels == [file_id,inst_id,pitch,octave]
        for file_id in train_dic.keys():
            labels = train_dic[file_id][1]
            signal = train_dic[file_id][0]
            if (np.sum(np.isnan(signal)*1)>0):
                print('signal has nan',train_dic[file_id][2])
                rainbow
            if (np.sum(np.isinf(signal)*1)>0):
                print('signal has inf',train_dic[file_id][2])
                rainbow

            if signal.size>=n_sig_out:
                N_slices = int(np.floor(signal.size/n_sig_out))
                train_slices += N_slices

                signal_slice = torch.from_numpy(signal[:N_slices*n_sig_out]).view(N_slices,n_sig_out).type(torch.float)
                if style==0:
                    labels_t = torch.zeros(N_slices,3).type(torch.long)
                if style==1:
                    labels_t = torch.zeros(N_slices,4).type(torch.long)
                labels_t[:,0] = labels[1]
                labels_t[:,1] = labels[2]
                labels_t[:,2] = labels[3]
                if style==1:
                    labels_t[:,3] = labels[4]

                train_sig_slices.append(signal_slice)
                train_labels.append(labels_t)
            #else:
                #print('too short, discarded',train_dic[file_id][2])

        eval_dic = np.load(data_path+inst_ref[inst][0]+'_evaldic.npy').item()
        for file_id in eval_dic.keys():
            labels = eval_dic[file_id][1]
            signal = eval_dic[file_id][0]
            if (np.sum(np.isnan(signal)*1)>0):
                print('signal has nan',eval_dic[file_id][2])
                rainbow
            if (np.sum(np.isinf(signal)*1)>0):
                print('signal has inf',eval_dic[file_id][2])
                rainbow

            if signal.size>=n_sig_out:
                N_slices = int(np.floor(signal.size/n_sig_out))
                eval_slices += N_slices

                signal_slice = torch.from_numpy(signal[:N_slices*n_sig_out]).view(N_slices,n_sig_out).type(torch.float)
                if style==0:
                    labels_t = torch.zeros(N_slices,3).type(torch.long)
                if style==1:
                    labels_t = torch.zeros(N_slices,4).type(torch.long)
                labels_t[:,0] = labels[1]
                labels_t[:,1] = labels[2]
                labels_t[:,2] = labels[3]
                if style==1:
                    labels_t[:,3] = labels[4]

                eval_sig_slices.append(signal_slice)
                eval_labels.append(labels_t)
            #else:
                #print('too short, discarded',eval_dic[file_id][2])

        test_dic = np.load(data_path+inst_ref[inst][0]+'_testdic.npy').item()
        for file_id in test_dic.keys():
            labels = test_dic[file_id][1]
            signal = test_dic[file_id][0]
            if (np.sum(np.isnan(signal)*1)>0):
                print('signal has nan',test_dic[file_id][2])
                rainbow
            if (np.sum(np.isinf(signal)*1)>0):
                print('signal has inf',test_dic[file_id][2])
                rainbow

            if signal.size>=n_sig_out:
                N_slices = int(np.floor(signal.size/n_sig_out))
                test_slices += N_slices

                signal_slice = torch.from_numpy(signal[:N_slices*n_sig_out]).view(N_slices,n_sig_out).type(torch.float)
                if style==0:
                    labels_t = torch.zeros(N_slices,3).type(torch.long)
                if style==1:
                    labels_t = torch.zeros(N_slices,4).type(torch.long)
                labels_t[:,0] = labels[1]
                labels_t[:,1] = labels[2]
                labels_t[:,2] = labels[3]
                if style==1:
                    labels_t[:,3] = labels[4]

                test_sig_slices.append(signal_slice)
                test_labels.append(labels_t)
            #else:
                #print('too short, discarded',test_dic[file_id][2])

        print('instrument train/eval/test',train_slices,eval_slices,test_slices)

    train_sig_slices = torch.cat(train_sig_slices,dim=0)
    train_labels = torch.cat(train_labels,dim=0)

    eval_sig_slices = torch.cat(eval_sig_slices,dim=0)
    eval_labels = torch.cat(eval_labels,dim=0)

    test_sig_slices = torch.cat(test_sig_slices,dim=0)
    test_labels = torch.cat(test_labels,dim=0)

    N_train = train_sig_slices.shape[0]
    N_eval = eval_sig_slices.shape[0]
    N_test = test_sig_slices.shape[0]

    train_dataset = torch.utils.data.TensorDataset(train_sig_slices,train_labels)
    eval_dataset = torch.utils.data.TensorDataset(eval_sig_slices,eval_labels)
    test_dataset = torch.utils.data.TensorDataset(test_sig_slices,test_labels)

    print('N_train/N_eval/N_test  ',N_train,N_eval,N_test)

    return [train_dataset,eval_dataset,test_dataset],[N_train,N_eval,N_test]
