"""Checks exact gradients against numerical gradients for
all currently available kernels."""
import unittest
#All currently available kernels are listed as keys in this dict.
from xGPR.kernels import KERNEL_NAME_TO_CLASS
from .kernel_specific_gradient_test import run_kernelspecific_test

class CheckKernelGradients(unittest.TestCase):
    """Checks the NMLL gradients for all currently implemented
    kernels."""

    def test_kernel_gradients(self):
        for kernel_name in KERNEL_NAME_TO_CLASS.keys():
            is_conv_kernel = "conv" in kernel_name.lower() or "graph" \
                    in kernel_name.lower()
            costcomps = run_kernelspecific_test(kernel_name,
                        conv_kernel = is_conv_kernel)
            for costcomp in costcomps:
                self.assertTrue(costcomp)

            #For convolution kernels, also test that graph or sequence averaging
            #works.
            if is_conv_kernel and "TwoLayer" not in kernel_name:
                print("****Testing with averaging****")
                costcomps = run_kernelspecific_test(kernel_name,
                        conv_kernel = is_conv_kernel,
                        averaging = 'full')
                for costcomp in costcomps:
                    self.assertTrue(costcomp)



if __name__ == "__main__":
    unittest.main()
