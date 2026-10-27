import torch
import torch.nn as nn

"""
PyTorch port of `labs/11-2_Visualization_Style-Transfer/guided_backprop.py`.

Reference (same as the original TF file):
https://github.com/eclique/keras-gradcam with adaption to tensorflow 2.0
https://github.com/nguyenhoa93/cnn-visualization-keras-tf2
Guided backprop itself: Springenberg et al., "Striving for Simplicity" (2015).

Guided backprop's ReLU rule is `grad_in = (x > 0) * (dy > 0) * dy`. A plain
ReLU's backward already computes `(x > 0) * dy` (that's what
`register_full_backward_hook` hands us as `grad_input`), so the *only* extra
thing guided backprop needs is clamping that already-computed grad_input to
be non-negative -- there's no need for a custom autograd Function / TF's
`@tf.custom_gradient`, just a backward hook on every `nn.ReLU`.

The original locates the "last 4D-output layer" using Keras' static
`layer.output_shape` (known without running data, since Keras builds a
static graph). PyTorch modules don't know their output shape without an
actual forward pass, so instead we auto-detect it by running one real batch
of `images` through the model with forward hooks attached -- which is also
strictly more robust (works for any model, not just ones with static shapes).
"""


class GuidedBackprop:
    def __init__(self, model, layer_name=None):
        self.model = model.eval()
        self.layer_name = layer_name
        self._target_module = None
        self._relu_hooks = []
        self._patch_guided_relu()
        if layer_name is not None:
            self._target_module = dict(self.model.named_modules())[layer_name]

    def _patch_guided_relu(self):
        def backward_hook(module, grad_input, grad_output):
            return (grad_input[0].clamp(min=0),)

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False  # register_full_backward_hook conflicts with inplace ops
                self._relu_hooks.append(module.register_full_backward_hook(backward_hook))

    def find_target_layer(self, example_input):
        """Auto-detect the last leaf submodule that produces a 4D (N,C,H,W)
        feature map, the PyTorch analogue of scanning `reversed(model.layers)`
        for `len(layer.output_shape) == 4`."""
        last_name, last_module = None, None

        def probe(name):
            def hook(_module, _inp, out):
                nonlocal last_name, last_module
                if isinstance(out, torch.Tensor) and out.dim() == 4:
                    last_name, last_module = name, _module
            return hook

        handles = []
        for name, module in self.model.named_modules():
            if next(module.children(), None) is None:  # leaf modules only
                handles.append(module.register_forward_hook(probe(name)))
        with torch.no_grad():
            self.model(example_input)
        for h in handles:
            h.remove()

        if last_module is None:
            raise ValueError("Could not find 4D layer. Cannot apply Guided Backpropagation")
        self.layer_name = last_name
        self._target_module = last_module

    def guided_backprop(self, images):
        """Guided Backpropagation method for visualizing input saliency."""
        if self._target_module is None:
            self.find_target_layer(images)

        images = images.clone().detach().requires_grad_(True)
        captured = {}

        def capture(_module, _inp, out):
            captured['value'] = out

        handle = self._target_module.register_forward_hook(capture)
        self.model.zero_grad(set_to_none=True)
        self.model(images)
        handle.remove()

        target = captured['value']
        grads, = torch.autograd.grad(target, images, grad_outputs=torch.ones_like(target))
        return grads

    def close(self):
        """Remove the ReLU hooks (the guided-backprop patch is otherwise permanent)."""
        for h in self._relu_hooks:
            h.remove()
        self._relu_hooks = []
