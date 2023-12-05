#%%

#updates package resources to handle version changes
import importlib, pkg_resources

importlib.reload(pkg_resources)

import tensorflow as tf
import tensorflow_quantum as tq
import cirq
import sympy
import numpy as np
import seaborn as sns
import collections

import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[...,
                                                           np.newaxis] / 255.0

print("Number of original training examples: ", len(x_train))
print("Number of original test examples: ", len(x_test))



def filter36(x, y):
    keep = (y == 3) | (y == 6)
    x, y = x[keep], y[keep]
    y = y == 3
    return x, y


x_train, y_train = filter36(x_train, y_train)
x_test, y_test = filter36(x_test, y_test)


print("Number of filtered training examples:", len(x_train))
print("Number of filtered test examples:", len(x_test))

print(y_train[0])
plt.imshow(x_train[0, :, :, 0])
plt.colorbar()


# %%
x_train_small = tf.image.resize(x_train, (4,4)).numpy()
x_test_small = tf.image.resize(x_test, (4,4)).numpy()

print(y_train[0])

plt.imshow(x_train_small[0,:,:,0], vmin=0, vmax=1)
plt.colorbar()

# %%
def remove_contra(xs, ys):
    mapping = collections.defaultdict(set)
    orig_x = {}
    
    for x,y in zip(xs, ys):
        orig_x[tuple(x.flatten())] = x
        mapping[tuple(x.flatten())].add(y)
    
    new_x = []
    new_y = []
    for flatten_x in mapping:
        x = orig_x[flatten_x]
        labels = mapping[flatten_x]
        if len(labels) == 1:
            new_x.append(x)
            new_y.append(next(iter(labels)))
    num3s = sum(1 for value in mapping.values() if len(value) == 1 and True in value)
    num6s = sum(1 for value in mapping.values() if len(value) == 1 and False in value)
    num_both = sum(1 for value in mapping.values() if len(value) == 2)
    print("Number of unique images:", len(mapping.values()))
    print("Number of unique 3s: ", num3s)
    print("Number of unique 6s: ", num6s)
    print("Number of unique contradicting labels (both 3 and 6): ", num_both)
    print()
    print("Initial number of images: ", len(xs))
    print("Remaining non-contradicting unique images: ", len(new_x))

    return np.array(new_x), np.array(new_y)

x_train_nocon, y_train_nocon = remove_contra(x_train_small, y_train)

# %%
THRESHOLD = 0.5

x_train_bin = np.array(x_train_nocon > THRESHOLD, dtype=np.float32)
x_test_bin = np.array(x_test_small > THRESHOLD, dtype=np.float32)

def convert_to_circuit(image):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit


x_train_circ = [convert_to_circuit(x) for x in x_train_bin]
x_test_circ = [convert_to_circuit(x) for x in x_test_bin]

SVGCircuit(x_train_circ[0])
# %%
x_train_tfcirc = tq.convert_to_tensor(x_train_circ)
x_test_tfcirc = tq.convert_to_tensor(x_test_circ)

class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)
demo_builder = CircuitLayerBuilder(data_qubits=cirq.GridQubit.rect(4,1), readout=cirq.GridQubit(-1,-1))

circuit = cirq.Circuit()
demo_builder.add_layer(circuit, gate=cirq.XX, prefix='xx')

SVGCircuit(circuit)
# %%
def create_quantum_model():
    data_qubits = cirq.GridQubit.rect(4,4);
    readout = cirq.GridQubit(-1,-1)
    circuit = cirq.Circuit()
    
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))
    builder = CircuitLayerBuilder(data_qubits=data_qubits, readout=readout)
    
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")
    
    circuit.append(cirq.H(readout))
    
    return circuit, cirq.Z(readout)

model_circuit, model_readout = create_quantum_model()


# %%
# Build the Keras model.
model = tf.keras.Sequential([
    # The input is the data-circuit, encoded as a tf.string
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    # The PQC layer returns the expected value of the readout gate, range [-1,1].
    tq.layers.PQC(model_circuit, model_readout),
])

y_train_hinge = 2.0*y_train_nocon-1.0
y_test_hinge = 2.0*y_test-1.0

def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)
model.compile(
    loss=tf.keras.losses.Hinge(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[hinge_accuracy]
)
print(model.summary())
# %%
EPOCHS=3
BATCH_SIZE=32
NUM_EXAMPLES=len(x_test_tfcirc)

x_train_tfcirc_sub = x_train_tfcirc[:NUM_EXAMPLES]
y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]

qnn_history = model.fit(
      x_train_tfcirc_sub, y_train_hinge_sub,
      batch_size=32,
      epochs=EPOCHS,
      verbose=1,
      validation_data=(x_test_tfcirc, y_test_hinge))

qnn_results = model.evaluate(x_test_tfcirc, y_test)
# %%
