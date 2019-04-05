from seq2seq_translation_tutorial import EncoderRNN,AttnDecoderRNN
import pickle
import torch
import numpy as np
# import Encoder
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker


class Generator:
    def __init__(self, encoder, decoder, comment_dict, SOS_token, EOS_token, device, max_length = 20):
        self.dict = {}
        for key,value in comment_dict.items():
            self.dict[value] = key
        self.device = device
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.max_length = max_length
    
    def generate(self, l):
        res = ""
        for idx in l:
            res += self.dict[idx] + " "
        return res
        

    def __call__(self, _input, _target):
        with torch.no_grad():
            encoder_hidden = self.encoder.initHidden().to(self.device)
            # Put the minibatch data in CUDA Tensors and run on the GPU if supported

            input_tensor = torch.LongTensor(_input).to(self.device)
            target_tensor = torch.LongTensor(_target).to(self.device)
            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            encoder_outputs = torch.zeros(input_length, self.encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = torch.tensor([[self.SOS_token]], device=self.device)
            decoder_hidden = encoder_hidden

            idx = self.SOS_token
            res = self.dict[idx] + ' '
            max_length = self.max_length
            # Teacher forcing: Feed the target as the next input
            # for di in range(target_length):
            while idx != self.EOS_token and max_length > 0:
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input.long(), decoder_hidden, encoder_outputs)
                prob = decoder_output[0].numpy().tolist()
                idx = prob.index(max(prob))
                word = self.dict[idx]
                res+= word + ' '
                decoder_input = torch.tensor([[idx]], device=self.device)
                max_length -= 1
                # decoder_input = target_tensor[di]  # Teacher forcing
            if max_length == 0:
                res += self.dict[self.EOS_token]
        return res

def load_model(type='attn'):
    with open(type+'_encoder1.ckpt', 'rb') as pfile:
        encoder = torch.load(pfile)
    with open(type + '_decoder1.ckpt', 'rb') as pfile:
        decoder = torch.load(pfile)
    return encoder, decoder

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.detach().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    # ax.set_xticklabels([''] + input_sentence.split(' ') +
    #                    ['<EOS>'], rotation=90)
    # ax.set_yticklabels([''] + output_words)

    # # Show label at every tick
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig('attention_map.png')


def main():
    # computing device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # load dict
    with open('data/comment_dict.pkl', 'rb') as pfile:
        comment_dict = pickle.load(pfile)
    SOS_token = comment_dict['<SOS>']
    EOS_token = comment_dict['<EOS>']
    # load test data
    with open('data/test_code_in_num.pkl', 'rb') as pfile:
        test_code_in_num = pickle.load(pfile)
    with open('data/test_comment_in_num.pkl', 'rb') as pfile:
        test_comment_in_num = pickle.load(pfile)
    with open('data/test.pkl', 'rb') as pfile:
        test_code_comment = pickle.load(pfile)
    # load model
    encoder, decoder = load_model()

    
    generator = Generator(encoder, decoder, comment_dict, SOS_token, EOS_token, device)
    
    idx = 500
    _, _code = test_code_comment[idx]
    _input = test_code_in_num[idx]
    _target = test_comment_in_num[idx]
    _output = generator(_input,_target)
    print(generator.generate(_target))
    print(_output)
    showAttention(_code, _output, decoder.attn.weight)


if __name__ == '__main__':
    main()
