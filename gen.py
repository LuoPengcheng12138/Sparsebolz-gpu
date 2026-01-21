import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from lsing import lsing_model, device


def clamp_labels_bipolar(m, digit: int, input_node_num: int, label_node_num: int, label_class_num: int):
    """
    m: bipolar {-1,+1}
    将 label bits 固定为：对应 digit 的位为 +1，其它为 -1
    label bits 在 [input_node_num, input_node_num+label_node_num)
    """
    bs = m.shape[0]
    labels = -torch.ones((bs, label_node_num), device=m.device)

    # repeated one-hot: digit, digit+10, digit+20, ...
    for k in range(0, label_node_num, label_class_num):
        idx = k + digit
        if idx < label_node_num:
            labels[:, idx] = 1.0

    m[:, input_node_num:input_node_num + label_node_num] = labels
    return m


def save_samples(images01, out_dir, prefix="sample"):
    """
    images01: [B, 784] in {0,1} or [0,1]
    """
    os.makedirs(out_dir, exist_ok=True)
    B = images01.shape[0]
    for i in range(B):
        img = images01[i].reshape(28, 28)
        plt.imsave(os.path.join(out_dir, f"{prefix}_{i:03d}.png"), img, cmap="Greys", vmin=0, vmax=1)


def main():
    # ---- load trained model ----
    ckpt_path = "model.pth"
    obj = torch.load(ckpt_path, map_location=device)

    # 兼容两种保存方式：
    # 1) torch.save(model, path) -> obj is a full model
    # 2) torch.save(model.state_dict(), path) -> obj is a state_dict
    if isinstance(obj, lsing_model):
        model = obj.to(device)
    elif isinstance(obj, dict):
        model = lsing_model().to(device)
        model.load_state_dict(obj)
    else:
        raise TypeError(f"Unknown checkpoint type: {type(obj)}")

    model.eval()

    # ---- generation hyperparams ----
    n_samples_per_digit = 16
    beta_start = 0.0
    beta_end = 5.0
    beta_step = 0.125

    # 每个 beta 走多少 sweeps：你可以加大让图更“定型”
    sweeps_per_beta = 200

    out_root = "gen_out"
    os.makedirs(out_root, exist_ok=True)

    with torch.no_grad():
        for digit in range(10):
            # random init state
            m = model.create_m(batch_size=n_samples_per_digit)

            # clamp labels
            m = clamp_labels_bipolar(
                m,
                digit=digit,
                input_node_num=model.input_node_num,
                label_node_num=model.label_node_num,
                label_class_num=model.label_class_num,
            )

            # anneal beta from low to high (paper uses 0 -> 5 with step 0.125)
            betas = np.arange(beta_start, beta_end + 1e-9, beta_step)
            for beta in betas:
                m = model.construct(m, model.group_gen, sample_num=sweeps_per_beta, beta=beta)
                # labels are excluded by group_gen, but we re-clamp defensively (optional)
                m = clamp_labels_bipolar(
                    m,
                    digit=digit,
                    input_node_num=model.input_node_num,
                    label_node_num=model.label_node_num,
                    label_class_num=model.label_class_num,
                )

            # take visible image bits, convert bipolar -> {0,1}
            img01 = (m[:, :model.input_node_num] + 1.0) / 2.0
            out_dir = os.path.join(out_root, f"digit_{digit}")
            save_samples(img01.cpu().numpy(), out_dir, prefix=f"d{digit}")

            print(f"[OK] digit {digit} saved to {out_dir}")


if __name__ == "__main__":
    main()
