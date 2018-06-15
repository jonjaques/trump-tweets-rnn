from textgenrnn import textgenrnn

tgen = textgenrnn(
  name='trump_char1'
)


# num_epochs: Number of epochs to train for (default: 50)
# gen_epochs: Number of epochs to run between generating sample outputs; good for measuring model progress (default: 1)
# batch_size: Batch size for training; may want to increase if running on a GPU for faster training (default: 128)
# train_size: Random proportion of sequence samples to keep: good for controlling overfitting. The rest will be used to train as the validation set. (default: 1.0/all). To disable training on the validation set (for speed), set validation=False.
# dropout: Random number of tokens to ignore each epoch. Good for controlling overfitting/making more resilient against typos, but setting too high will cause network to converge prematurely. (default: 0.0)
# is_csv: Use with train_from_file if the source file is a one-column CSV (e.g. an export from BigQuery or Google Sheets) for proper quote/newline escaping.
tgen.reset()
tgen.train_from_file('data/trump.txt', 
  new_model=True, 
  num_epochs=3, 
  batch_size=128,
  train_size=0.8,
  dropout=0.5,
  word_level=False, 
  rnn_bidirectional=False, 
  rnn_size=256, 
  max_length=60)