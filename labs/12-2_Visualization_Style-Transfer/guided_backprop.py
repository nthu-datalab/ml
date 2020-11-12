import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

"""
Reference: 
https://github.com/eclique/keras-gradcam with adaption to tensorflow 2.0  
https://github.com/nguyenhoa93/cnn-visualization-keras-tf2
"""

@tf.custom_gradient
def GuidedRelu(x):
    def grad(dy):
        gate_f = tf.cast(x > 0, "float32")
        gate_R = tf.cast(dy > 0,"float32")
        return gate_f * gate_R * dy
    return tf.nn.relu(x), grad

class GuidedBackprop:
    def __init__(self, model, layerName=None):
        self.model = model
        self.layerName = layerName
        if self.layerName == None:
            self.layerName = self.find_target_layer()

        self.gbModel = self.build_guided_model()
        
    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply Guided Backpropagation")

    def build_guided_model(self):
        gbModel = Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layerName).output]
        )
        layer_dict = [layer for layer in gbModel.layers[1:] if hasattr(layer, "activation")]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = GuidedRelu
        return gbModel
    
    @tf.function()
    def guided_backprop(self, images):
        """Guided Backpropagation method for visualizing input saliency."""
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            outputs = self.gbModel(inputs)
        grads = tape.gradient(outputs, inputs)
        return grads