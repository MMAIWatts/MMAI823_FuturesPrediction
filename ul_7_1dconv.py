import pandas as pd
from keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, GlobalMaxPool1D
from keras.models import Model

# Local variables
train_path = 'data/train_set.csv'
test_path = 'data/test_set.csv'
randomState = 42

# read DataFrame
df = pd.read_csv(train_path, index_col=0)
df_test = pd.read_csv(test_path, index_col=0)

# separate labels
X_train = df[df.columns[:-16]]
X_test = df_test[df_test.columns[:-16]]
y_trains = df[df.columns[-16:]]
y_tests = df_test[df.columns[-16:]]

#  create input layer for time window of 20 days, width of 26 features
input_seq = Input(shape=(20, 26))
# number of convolutional filters
nb_filters = 10
convolved = Conv1D(nb_filters, 8, padding='same', activation='tanh')(input_seq)
processed = GlobalMaxPool1D()(convolved)
compressed = Dense(50, activation='tanh')(processed)
compressed = Dropout(0.3)(compressed)
model = Model(inputs=input_seq, outputs=compressed)

