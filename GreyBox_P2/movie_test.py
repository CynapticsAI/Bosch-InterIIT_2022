import tensorflow as tf
import tensorflow_hub as hub
import numpy
import torch

hub_url = "https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3"

encoder = hub.KerasLayer(hub_url, trainable=True)

inputs = tf.keras.layers.Input(shape=[None, None, None, 3],dtype=tf.float32,name='image')

# [batch_size, 600]
outputs = encoder(dict(image=inputs))

model = tf.keras.Model(inputs, outputs, name='movinet')

def get_probabs(pt_tensor,device): #Input(batch_sz,32,3,64,64)

  #Requires (batch_sz,32,h,w,3)
	pt_tensor = pt_tensor.permute(0,1,4,3,2)
	example_input = tf.convert_to_tensor(pt_tensor.to('cpu').numpy())
	example_output = model(example_input)
	example_output = tf.nn.softmax(example_output)
	class_label = torch.from_numpy(tf.argmax(example_output, axis = 1).numpy()).to(device=device)
	return(class_label)
