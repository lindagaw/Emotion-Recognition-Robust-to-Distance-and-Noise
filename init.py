from colorama import Fore, Back, Style

sample_rate = 44100
frame_number = 48
hop_length = 441  # frame size= 2 * hop
segment_length = int(sample_rate * 0.2)  # 0.2
segment_pad = int(sample_rate * 0.02)  # 0.02
overlapping = int(sample_rate * 0.1)  # 0.1
NumofFeaturetoUse = 272  # int(sys.argv[1])

print('Please specify the' + Fore.YELLOW + ' number of classes ' +
      Style.RESET_ALL + 'in your training:')
classes = int(input())

print('Please specify the number of most significant features to use:')
NumofFeaturetoUse = int(input())

print('Please specify the number of neurons in the dense layer(s).')
n_neurons = int(input())

print('Please specify the number of dense layer(s).')
dense_layers = int(input())

print('Please specify the number of conv layer(s).')
num_layers = int(input())

print('Please specify the size of a kernal in each conv layer.')
fillength = int(input())

print('Please specify the number of kernals in each conv layer.')
nbindex = int(input())

print('Please specify the dropout parameter in applicable layers.')
dropout = float(input())

print('Please specify the size of each batch during training.')
n_batch = int(input())

print('Please specify the number of epoches during training.')
n_epoch = int(input())
