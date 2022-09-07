import jax
import GPUtil
import os


def jax_setup(device="auto", x64=False, n_gpu=1):
    if device.lower() == "auto":
        device = "cpu" if len(GPUtil.getGPUs()) == 0 else "gpu"

    if device.lower() == "cpu":
        jax.config.update("jax_platform_name", "cpu")
    elif device.lower() == "gpu":
        devices = GPUtil.getAvailable(limit=float("inf"))
        devices = [str(x - 1) for x in devices]
        print(f"Available Devices: {','.join(devices)}")
        if len(devices) < n_gpu:
            raise Exception(
                f"Insufficient GPUs (requested {n_gpu}, only {len(devices)} available."
            )
        else:
            print(f"Using Devices: {','.join(devices[:n_gpu])}")
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices[:n_gpu])
            assert len(jax.devices()) == n_gpu

    if x64:
        jax.config.update("jax_enable_x64", True)
