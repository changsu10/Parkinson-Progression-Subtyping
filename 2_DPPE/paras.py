
class hyperParas:
    def __init__(self):
        sequence_length = [10, 12, 14, 16]
        embed_size = [None]
        hidden_size = [16, 24, 32, 48]
        num_layers = [1]
        batch_size = [1]
        LR = [0.005, 0.001]

        optimizer = ['Adam']

        self.paras = {'sequence_length': sequence_length,
                      'embed_size': embed_size, 'hidden_size': hidden_size, 'num_layers': num_layers,
                      'batch_size': batch_size, 'LR': LR, 'optimizer': optimizer}

    def __getParas__(self):
        return self.paras
