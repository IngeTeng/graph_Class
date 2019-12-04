import os

from args import *
from train_test_functions import *
from model import *
from data import *

args = Args(cuda=torch.cuda.is_available(), graph_name='ENZYMES')
#args = Args(cuda=torch.cuda.is_available(), graph_name='BeetleFly')

args.epochs = 2000
args.batch_size = 128
args.reco_importance = 0.1
args.loss = nn.BCELoss()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
print('CUDA', args.cuda)

graphs = graph_load_batch(data_directory=args.data_directory, name=args.graph_name)

dataloaders_train, dataloaders_test = create_loaders(graphs, args)

results = {}

args.num_fold = None

for i in range(10):

    print('Fold number: {:.0f}'.format(i + 1))
    args.num_fold = i

    rnn_embedding = RecurrentEmbedding(input_size=args.node_dim,
                                       embedding_size=args.embedding_size_rnn,
                                       hidden_size=args.hidden_size_rnn,
                                       num_layers=args.num_layers,
                                       is_cuda=args.cuda)

    var = VAR(h_size=args.hidden_size_rnn,
              embedding_size=args.embedding_size_output,
              y_size=args.node_dim,
              is_cuda=args.cuda)

    rnn_classifier = RecurrentClassifier(input_size=args.hidden_size_rnn,
                                         embedding_size=args.embedding_size_rnn,
                                         hidden_size=args.hidden_size_rnn,
                                         num_layers=args.num_layers,
                                         num_class=args.num_class,
                                         is_cuda=args.cuda)

    if args.cuda:
        rnn_embedding = rnn_embedding.cuda()
        var = var.cuda()
        rnn_classifier = rnn_classifier.cuda()

    learning_accuracy_test = classifier_train(args,
                                              dataloaders_train[i],
                                              dataloaders_test[i],
                                              rnn_embedding, var, rnn_classifier)

    accuracy_test, scores, predicted_labels, true_labels, vote = vote_test(args,
                                                                           rnn_embedding,
                                                                           var,
                                                                           rnn_classifier,
                                                                           dataloaders_test[i],
                                                                           num_iteration=100)

    results[i] = {'rnn': rnn_embedding, 'output': var, 'classifier_1': rnn_classifier,
                  'acc_test': accuracy_test, 'scores': scores}

print([results[r]['acc_test'] for r in results])

print(np.mean([results[r]['acc_test'] for r in results]),
      np.std([results[r]['acc_test'] for r in results]))