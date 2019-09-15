class Config(object):
    ### This hyper-parameter is about the data
    data_path = '../data/'
    granularity = 500.10000020
    train_file_raw = '../data/train_raw_lower'
    bpe_model = '../data/bpe_500.model'
    subwords_file = '../data/bpe_500_subword.pkl'
    words_file = '../data/words_100000.pkl'
    logfile = '../log/500_fix_loss'
    ### This hyper-parameter is about the LSTM
    vocab_size = 100010
    cell_clips = 3
    use_skip_connections = True
    num_layers = 3
    proj_clip = 3
    projection_dim = 512
    dim = 4096
    drop_out_p = 0.1
    ### This hyper-parameter is about the CNN(subword)
    subword_embedding_size = 64
    subword_fliter = [[1, 64], [2, 64], [3, 128], [4, 256], [5, 512]]
    n_subword_fliter = 1024
    n_subwords = 510
    max_subwords_per_word = 7
    cnn_activation = torch.nn.ReLU
    ### This hyper-parameter is about the Highway
    n_highway = 2
    ### This hyper-parameter is about the training and saving model
    mode = 'train'
    reg_lambda = 0.0001
    lr = 0.033
    use_gpu = True
    start_epoch = 2
    epoch = 10
    batch_size = 80
    batch_size_test = 40
    random_shuffle = True
    max_length = 20 # The maxlenght of the sentence
    n_batch = 6
    model_prefix = '../model/'
    optim_path = None # '/home/lijt/data/pytorch_bpe/model/epoch_6_main_bpe_opt_500.1000002.pth_tmp'
    model_path = '/home/lijt/data/pytorch_bpe/model/epoch_1_main_bpe_500.1000002.pth'
