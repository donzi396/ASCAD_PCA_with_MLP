from ASCAD_train_models import load_ascad
from ASCAD_test_models import full_ranks
import numpy as np
from sklearn.decomposition import PCA
import sys
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint


def calc_pca(X_profiling,X_attack,n_components):
	pca = PCA(n_components=n_components)
	pca_X_profiling = pca.fit_transform(X_profiling)
	pca_X_attack = pca.transform(X_attack)
	#return sum(pca.explained_variance_ratio_)
	return pca_X_profiling, pca_X_attack

def build_mlp(input_dim,node=200,layer_nb=6):
	model = Sequential()
	model.add(Dense(node, input_dim=input_dim, activation='relu'))
	for i in range(layer_nb-2):
		model.add(Dense(node, activation='relu'))
	model.add(Dense(256, activation='softmax'))
	optimizer = RMSprop(lr=0.00001)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

def train_mlp(X_profiling, Y_profiling, model, label, epochs=150, batch_size=100):
	model_name = 'my_mlp_components' + label + '_best_desync0_epochs' + str(epochs) + '_batchsize' + str(batch_size) + '.h5'
	save_file_name =  'ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/' + model_name
	save_model = ModelCheckpoint(save_file_name)
	callbacks=[save_model]
	# Get the input layer shape
	input_layer_shape = model.get_layer(index=0).input_shape
	# Sanity check
	if input_layer_shape[1] != len(X_profiling[0]):
		print("Error: model input shape %d instead of %d is not expected ..." % (
		input_layer_shape[1], len(X_profiling[0])))
		sys.exit(-1)
	else:
		print("training model (input shape: %d) ..." % input_layer_shape[1])

	model.fit(x=X_profiling, y=to_categorical(Y_profiling, num_classes=256), batch_size=batch_size,
						verbose=0, epochs=epochs, callbacks=callbacks)

def test_mlp(X_attack, Metadata_attack, model, label, num_traces=2000):
	# We test the rank over traces of the Attack dataset, with a step of 10 traces
	ranks = full_ranks(model, X_attack, Metadata_attack, 0, num_traces, 10)
	# We plot the results
	x = [ranks[i][0] for i in range(0, ranks.shape[0])]
	y = [ranks[i][1] for i in range(0, ranks.shape[0])]
	plt.plot(x, y, label=label + ' components')

if __name__ == "__main__":
	ascad_data_folder = 'ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/'
	for desync in [0,50,100]:
		if desync == 0:
			ascad_database = ascad_data_folder + 'ASCAD.h5'
		elif desync == 50:
			ascad_database = ascad_data_folder + 'ASCAD_desync50.h5'
		elif desync == 100:
			ascad_database = ascad_data_folder + 'ASCAD_desync100.h5'
		(X_profiling, Y_profiling), (X_attack, Y_attack), (Metadata_profiling, Metadata_attack) = load_ascad(ascad_database, load_metadata=True)
		for epochs in [100,200,300,400,500]:
			for scale in ['large','small']:
				if scale == 'large':
					my_range = range(700,0,-100)
				elif scale == 'small':
					my_range = range(90,0,-10)
				plt.title('PCA performance, ' + scale + ', desync' + str(desync) + ', ' + str(epochs) + ' epochs')
				plt.xlabel('number of traces')
				plt.ylabel('rank')
				plt.grid(True)
				for n_components in my_range:
					pca_X_profiling, pca_X_attack = calc_pca(X_profiling, X_attack, n_components)
					model = build_mlp(n_components)
					train_mlp(pca_X_profiling, Y_profiling, model, str(n_components), epochs=epochs)
					test_mlp(pca_X_attack, Metadata_attack, model, str(n_components))
				plt.legend()
				plt.savefig('PCA_'+scale+'_desync'+str(desync)+'_epochs'+str(epochs)+'.png')
				plt.figure()