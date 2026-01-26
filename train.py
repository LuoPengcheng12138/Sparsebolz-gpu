import torch
from download import download
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_size = 1000
test_size = 100

from read_MNIST import decode_idx3_ubyte as read_images
from read_MNIST import decode_idx1_ubyte as read_labels
images = torch.tensor(read_images("./MNIST_Data/train/train-images-idx3-ubyte")).reshape(60000, 28*28).to(device)
labels = torch.tensor(read_labels("./MNIST_Data/train/train-labels-idx1-ubyte")).to(device)
images = images[:train_size]
labels = labels[:train_size]
train_dataset = torch.utils.data.TensorDataset(images, labels)
train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True, pin_memory=False)

images = torch.tensor(read_images("./MNIST_Data/test/t10k-images-idx3-ubyte")).reshape(10000, 28*28).to(device)
labels = torch.tensor(read_labels("./MNIST_Data/test/t10k-labels-idx1-ubyte")).to(device)
images = images[:test_size]
labels = labels[:test_size]
test_dataset = torch.utils.data.TensorDataset(images, labels)
test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=True, pin_memory=False)


from tqdm import tqdm
from lsing import lsing_model
import numpy as np

input_node_num=28*28
label_class_num=10
label_node_num=50
all_node_num=4264

import torch
from tqdm import tqdm
from lsing import lsing_model
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- NMFA hyperparams (建议按 phase 分开) --------
NMFA_POS_STEPS   = 400
NMFA_POS_T_START = 2.0
NMFA_POS_T_END   = 1.0
NMFA_POS_SIGMA   = 0.05
NMFA_POS_ALPHA   = 0.85

NMFA_NEG_STEPS   = 600
NMFA_NEG_T_START = 5.0
NMFA_NEG_T_END   = 0.8
NMFA_NEG_SIGMA   = 0.15
NMFA_NEG_ALPHA   = 0.85

NMFA_INF_STEPS   = 300
NMFA_INF_T_START = 2.0
NMFA_INF_T_END   = 1.0
NMFA_INF_SIGMA   = 0.05
NMFA_INF_ALPHA   = 0.85

# 每个 phase 的采样次数：越大相关估计越稳，但更慢
N_DRAWS_POS = 8
N_DRAWS_NEG = 8

def predict_labels_soft(s, model):
    """直接用连续 s 做分类读出（更稳，不需要再采样）。"""
    if getattr(model, "label_class_nodes", None):
        class_keys = sorted(model.label_class_nodes.keys())
        class_nodes = [model.label_class_nodes[k] for k in class_keys]
        scores = torch.stack([s[:, idxs].sum(dim=1) for idxs in class_nodes], dim=1)
        return scores.argmax(dim=1)

    # 若没有 label_class_nodes，就退回到你的老布局（一般你有）
    input_node_num = model.input_node_num
    label_node_num = model.label_node_num
    label_class_num = model.label_class_num
    logits = s[:, input_node_num:input_node_num+label_node_num].reshape(
        -1, label_node_num//label_class_num, label_class_num
    )
    return logits.sum(dim=-2).argmax(dim=-1)

def expand_draws_to_samples(m_draws):
    """
    m_draws: [B, n_draws, N] -> [B*n_draws, N]
    """
    B, D, N = m_draws.shape
    return m_draws.reshape(B * D, N)

@torch.no_grad()
def main():
    model = lsing_model(label_class_num=label_class_num, label_node_num=label_node_num).to(device)

    # ---- persistent negative continuous state (PCD-style) ----
    persistent_s_neg = None  # [B,N] float in [-1,1]

    test_acc = []
    for epoch in range(5):
        acc = torch.tensor([], device=device)
        bar = tqdm(train_dataset_loader)

        for images_batch, labels_batch in bar:
            bs = images_batch.shape[0]

            # visible nodes (image + label)
            visible_nodes = model._get_visible_nodes()
            if visible_nodes is None:
                raise ValueError("visible_nodes is None")

            # =======================
            # Positive phase (NMFA, clamped to image+label)
            # =======================
            m_pos0 = model.create_m(images_batch, labels_batch)  # bipolar, visible 已被 clamp :contentReference[oaicite:2]{index=2}

            s_pos = model.nmfa_warm_start(
                m_pos0,
                steps=NMFA_POS_STEPS,
                T_start=NMFA_POS_T_START,
                T_end=NMFA_POS_T_END,
                sigma=NMFA_POS_SIGMA,
                alpha=NMFA_POS_ALPHA,
                clamp_nodes=visible_nodes,      # 正相：固定 image+label
                return_continuous=True,
            )

            m_pos_draws = model.sample_bipolar_from_soft(
                s_pos,
                clamp_nodes=visible_nodes,
                clamp_values=m_pos0,
                n_draws=N_DRAWS_POS,
            )  # [B, D, N]
            m_data = expand_draws_to_samples(m_pos_draws)        # [B*D, N]

            # =======================
            # Negative phase (NMFA, free run, persistent)
            # =======================
            if persistent_s_neg is None or persistent_s_neg.shape[0] != bs:
                # 初始 persistent state：用随机 bipolar 更像原始链
                m_neg0 = model.create_m(batch_size=bs)           # bipolar random
                persistent_s_neg = m_neg0.to(torch.float32)

            s_neg = model.nmfa_warm_start(
                persistent_s_neg,
                steps=NMFA_NEG_STEPS,
                T_start=NMFA_NEG_T_START,
                T_end=NMFA_NEG_T_END,
                sigma=NMFA_NEG_SIGMA,
                alpha=NMFA_NEG_ALPHA,
                clamp_nodes=None,              # 负相：不 clamp
                return_continuous=True,
            )
            persistent_s_neg = s_neg.detach()

            m_neg_draws = model.sample_bipolar_from_soft(
                s_neg,
                clamp_nodes=None,
                clamp_values=None,
                n_draws=N_DRAWS_NEG,
            )
            m_model = expand_draws_to_samples(m_neg_draws)        # [B*D, N]

            # =======================
            # Parameter update (保持你原 updateParams 不改)
            # =======================
            model.updateParams(m_data, m_model, batch_size=m_data.shape[0])

            # =======================
            # Train-time inference (NMFA classify: clamp input only)
            # =======================
            m_inf0 = model.create_m(images_batch)  # 只 clamp input，label 随机 :contentReference[oaicite:3]{index=3}
            input_nodes = model.input_nodes
            s_inf = model.nmfa_warm_start(
                m_inf0,
                steps=NMFA_INF_STEPS,
                T_start=NMFA_INF_T_START,
                T_end=NMFA_INF_T_END,
                sigma=NMFA_INF_SIGMA,
                alpha=NMFA_INF_ALPHA,
                clamp_nodes=input_nodes,        # 推断：只固定 image bits
                return_continuous=True,
            )
            preds = predict_labels_soft(s_inf, model)

            logits = torch.where(preds == labels_batch, 1.0, 0.0)
            acc = torch.cat([acc, logits])
            bar.set_postfix({"acc": acc.mean().item()})

        # =======================
        # Test inference (NMFA classify)
        # =======================
        acc = torch.tensor([], device=device)
        for images_batch, labels_batch in test_dataset_loader:
            m_inf0 = model.create_m(images_batch)
            input_nodes = model.input_nodes
            s_inf = model.nmfa_warm_start(
                m_inf0,
                steps=NMFA_INF_STEPS,
                T_start=NMFA_INF_T_START,
                T_end=NMFA_INF_T_END,
                sigma=NMFA_INF_SIGMA,
                alpha=NMFA_INF_ALPHA,
                clamp_nodes=input_nodes,
                return_continuous=True,
            )
            preds = predict_labels_soft(s_inf, model)
            logits = torch.where(preds == labels_batch, 1.0, 0.0)
            acc = torch.cat([acc, logits])

        print(f"epoch {epoch} test result acc:{acc.mean().item()}")
        test_acc.append(acc.mean().item())
        torch.save(model, "model.pth")
        np.savetxt("test_acc.txt", test_acc)

if __name__ == "__main__":
    main()
