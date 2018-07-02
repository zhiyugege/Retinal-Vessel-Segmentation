class DefaultConfig(object):
    
    
    batch_size = 1
    use_GPU = True
    num_workers = 1
    print_freq = 20
    
    result_file = 'result.csv'
    max_epoch = 10000
    lr = 0.001
    lr_decay = 0.1
    weight_decay = 1e-4
    loss_weight = [0.1,0.8,0.1]
