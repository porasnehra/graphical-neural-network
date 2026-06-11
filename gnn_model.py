import tensorflow as tf

class SimpleGCNLayer(tf.keras.layers.Layer):
    """
    A pure TensorFlow implementation of a Graph Convolutional Network (GCN) layer.
    Math: Output = Activation( Adjacency_Matrix * (Features * Weights) )
    """
    def __init__(self, output_dim, activation='relu', **kwargs):
        super(SimpleGCNLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        feature_shape = input_shape[0]
        self.kernel = self.add_weight(
            shape=(feature_shape[-1], self.output_dim),
            initializer='glorot_uniform',
            name='kernel',
            trainable=True
        )
        super(SimpleGCNLayer, self).build(input_shape)

    def call(self, inputs):
        features, adjacency = inputs
        
        # Feature transformation (X * W)
        transformed_features = tf.matmul(features, self.kernel)
        
        # Message passing (A * XW)
        output = tf.matmul(adjacency, transformed_features)
        
        if self.activation is not None:
            output = self.activation(output)
            
        return output

def create_gnn_model(num_features):
    """Create GNN model with custom SimpleGCNLayer."""
    x_input = tf.keras.layers.Input(shape=(None, num_features), name="node_features")
    a_input = tf.keras.layers.Input(shape=(None, None), name="adjacency_matrix")

    gcn_1 = SimpleGCNLayer(32, activation='relu')([x_input, a_input])
    gcn_2 = SimpleGCNLayer(16, activation='relu')([gcn_1, a_input])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(gcn_2)
    
    model = tf.keras.Model(inputs=[x_input, a_input], outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
