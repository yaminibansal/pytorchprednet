savepath = '/home/ybansal/Documents/Research/pytorchprednet/Data/confused_ball/Results/'

for batch_size in [10, 50, 100]:
    for num_samples in [1, 5, 10]:
        for noise_dim in [5, 10, 20, 50]:
            for hidden_size in [20, 30, 40, 50, 100]:
                for rbf in [4.0, 5.0, 10.0, 15.0]:
                    fn = '_'.join([str(x) for x in [batch_size, num_samples, noise_dim, hidden_size, rbf]]).replace('.', '_')
                    string='''--dataset
ball
--train_root
/home/ybansal/Documents/Research/pytorchprednet/Data/confused_ball/train2000.hkl
--val_root
/home/ybansal/Documents/Research/pytorchprednet/Data/confused_ball/val500.hkl
--num_epochs
500
--batch_size
%d
--num_samples
%d
--modelname
StochLSTMFCEncDec
--hid_size
%d
--num_noise_dim
%d
--enc_int_layers
1000
2000
--dec_int_layers
2000
1000
--sig_rbf
%f
--lr
0.001
--savepath
%s
--num_inp_plts
2	
--num_gen_plts
5
'''%(batch_size, num_samples, hidden_size, noise_dim, rbf, savepath + fn )
                    string = string.encode()

                    with open('/home/ybansal/Documents/Research/pytorchprednet/prednet/stochastic/inputfiles/mmd_timeseries/hyperparam_search/input_'+fn+'.txt', 'w') as f:
                        f.write(string)
