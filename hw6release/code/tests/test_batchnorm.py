import random
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from .utils import LayerTest

from neural_networks.layers import BatchNorm1D

class TestBatchNorm1D(LayerTest):
    LayerCls = BatchNorm1D
    layer_config = {
        "momentum": 0.95,
        "eps": 1e-8,
        "n_in": None,
    }
    num_iters = 10

    def _test(self, mode="forward"):
        layer: BatchNorm1D = self.LayerCls(**self.layer_config)
        layer._init_parameters(self.test_data["inference_inputs"].shape)
        layer.parameters["gamma"] = self.test_data["gamma"]
        layer.parameters["beta"] = self.test_data["beta"]
        if mode == "forward":
            # this is for running tests in forward module in training mode
            for i in range(self.num_iters):
                input_data = self.test_data[str(i) + "input"]
                out = layer.forward(input_data, mode="train")
                running_mean = layer.cache["running_mu"]
                running_var = layer.cache["running_var"]
                ref_running_mean = self.test_data[str(i) + "running_mu"]
                ref_running_var = self.test_data[str(i) + "running_var"]
                assert_almost_equal(running_mean, ref_running_mean, decimal=4)
                assert_almost_equal(running_var, ref_running_var, decimal=4)

            # inference in test mode
            layer.parameters["gamma"] = self.test_data["gamma"]
            layer.parameters["beta"] = self.test_data["beta"]

            out = layer.forward(self.test_data["inference_inputs"], mode="test")
            assert_almost_equal(out, self.test_data["inference_output"], decimal=4)
        elif mode == "backward":
            layer.parameters["gamma"] = self.test_data["gamma"]
            layer.parameters["beta"] = self.test_data["beta"]
            layer.cache["X"] = self.test_data["X"]
            layer.cache["X_hat"] = self.test_data["X_hat"]
            layer.cache["mu"] = self.test_data["mu"]
            layer.cache["var"] = self.test_data["var"]
            Xgrad = layer.backward(self.test_data["upstream_grad_X"])
            assert_almost_equal(Xgrad, self.test_data["x_grad"], decimal=4)
            assert_almost_equal(layer.gradients["gamma"], self.test_data["gamma_grad"], decimal=4)
            assert_almost_equal(layer.gradients["beta"], self.test_data["beta_grad"], decimal=4)
            
        return True
    
    def test_forward(self):
        """
        Test forward pass of the layer.
        """
        return self._test(mode="forward")
    
    def test_backward(self):
        """
        Test backward pass of the layer.
        """
        return self._test(mode="backward")