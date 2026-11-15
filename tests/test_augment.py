import torch

from analog_geometry.common import augment_cifar


def test_shape_and_determinism():
    x = torch.randn(16, 3, 32, 32)
    a = augment_cifar(x, torch.Generator().manual_seed(0))
    b = augment_cifar(x, torch.Generator().manual_seed(0))
    assert a.shape == x.shape
    assert torch.equal(a, b)


def test_each_output_is_shifted_or_flipped_crop_of_input():
    x = torch.randn(8, 3, 32, 32)
    out = augment_cifar(x, torch.Generator().manual_seed(1))
    xp = torch.nn.functional.pad(x, (4, 4, 4, 4))
    for i in range(8):
        found = False
        for dy in range(9):
            for dx in range(9):
                crop = xp[i, :, dy:dy + 32, dx:dx + 32]
                if torch.equal(out[i], crop) or torch.equal(out[i], crop.flip(-1)):
                    found = True
        assert found
