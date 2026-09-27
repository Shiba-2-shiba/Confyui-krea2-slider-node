import unittest

import torch
from krea2_slider_node.quantization import FrozenLinear, quantize_convrot
from krea2_slider_node.lora import inject_lora, load_lora_state_dict, lora_state_dict


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm GPU required")
class GPUStagingTests(unittest.TestCase):
    def test_cpu_storage_and_resident_storage_have_same_multilayer_gradients(self):
        torch.manual_seed(28)
        codes, scale = quantize_convrot(torch.randn(64, 64), 64)
        cpu = torch.nn.Sequential(FrozenLinear(codes, scale=scale, group_size=64), torch.nn.SiLU(),
                                  FrozenLinear(codes.clone(), scale=scale.clone(), group_size=64))
        gpu = torch.nn.Sequential(FrozenLinear(codes.cuda(), scale=scale.cuda(), group_size=64), torch.nn.SiLU(),
                                  FrozenLinear(codes.cuda(), scale=scale.cuda(), group_size=64))
        inject_lora(cpu, rank=2, alpha=2, target="all", device="cuda")
        inject_lora(gpu, rank=2, alpha=2, target="all", device="cuda")
        with torch.no_grad():
            for module in (cpu[0], cpu[2]):
                module.lora_up.weight.normal_(std=0.01)
        load_lora_state_dict(gpu, lora_state_dict(cpu))
        x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, b = cpu(x), gpu(x)
        torch.testing.assert_close(a, b)
        a.float().square().mean().backward()
        b.float().square().mean().backward()
        torch.testing.assert_close(cpu[0].lora_down.weight.grad, gpu[0].lora_down.weight.grad)
        self.assertGreater(float(cpu[0].lora_down.weight.grad.abs().sum()), 0)
        self.assertEqual(cpu[0].base.weight.device.type, "cpu")
