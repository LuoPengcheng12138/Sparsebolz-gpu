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
from line_profiler import profile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_node_num = 28*28
label_class_num = 10
label_node_num = 50

# ---- MF hyperparams ----
MF_POS_STEPS = 80
MF_NEG_STEPS = 80
MF_INF_STEPS = 80

MF_BETA = 1.0
MF_DAMP = 0.5  # 0=纯固定点; 建议 0.3~0.7 更稳

def predict_labels_from_mu(mu, model):
    """从连续 mu 直接读出 label（不需要采样）。"""
    if getattr(model, "label_class_nodes", None):
        class_keys = sorted(model.label_class_nodes.keys())
        class_nodes = [model.label_class_nodes[k] for k in class_keys]
        scores = torch.stack([mu[:, idxs].sum(dim=1) for idxs in class_nodes], dim=1)
        return scores.argmax(dim=1)

    logits = mu[:, input_node_num:input_node_num+label_node_num].reshape(
        -1, label_node_num//label_class_num, label_class_num
    )
    return logits.sum(dim=-2).argmax(dim=-1)

@profile
def main():
    model = lsing_model(label_class_num=label_class_num, label_node_num=label_node_num).to(device)

    visible_nodes = model._get_visible_nodes()
    if visible_nodes is None:
        visible_nodes = torch.arange(model.input_node_num + model.label_node_num, device=device)

    # persistent MF state for negative phase (PCD-style but MF)
    persistent_mu_neg = None

    test_acc = []
    with torch.no_grad():
        for epoch in range(5):
            acc = torch.tensor([], device=device)
            bar = tqdm(train_dataset_loader)

            for images_batch, labels_batch in bar:
                bs = images_batch.shape[0]

                # ========= Positive phase (MF, clamp visible=image+label) =========
                m_pos = model.create_m(images_batch, labels_batch)  # bipolar {-1,+1}
                mu0_pos = m_pos.to(torch.float32)                   # init

                mu_pos = model.mf_relax(
                    mu0_pos,
                    steps=MF_POS_STEPS,
                    beta=MF_BETA,
                    alpha=MF_DAMP,
                    clamp_nodes=visible_nodes,
                    clamp_values=m_pos,   # clamp to true visible bits
                )

                # ========= Negative phase (MF, free) =========
                if persistent_mu_neg is None or persistent_mu_neg.shape[0] != bs:
                    m0 = model.create_m(batch_size=bs)  # random bipolar
                    persistent_mu_neg = m0.to(torch.float32)

                mu_neg = model.mf_relax(
                    persistent_mu_neg,
                    steps=MF_NEG_STEPS,
                    beta=MF_BETA,
                    alpha=MF_DAMP,
                    clamp_nodes=None,
                    clamp_values=None,
                )
                persistent_mu_neg = mu_neg.detach()

                # ========= Parameter update (MF-ELBO update) =========
                model.updateParamsMF(mu_pos, mu_neg, batch_size=bs)

                # ========= Inference / train accuracy (MF classify: clamp input only) =========
                m_inf = model.create_m(images_batch)  # clamp input only, label随机
                mu0_inf = m_inf.to(torch.float32)

                mu_inf = model.mf_relax(
                    mu0_inf,
                    steps=MF_INF_STEPS,
                    beta=MF_BETA,
                    alpha=MF_DAMP,
                    clamp_nodes=model.input_nodes,  # 只 clamp 输入像素
                    clamp_values=m_inf,
                )
                preds = predict_labels_from_mu(mu_inf, model)

                logits = torch.where(preds == labels_batch, 1.0, 0.0)
                acc = torch.cat([acc, logits])
                bar.set_postfix({"acc": acc.mean().item()})

            # ========= Test inference (MF classify) =========
            acc = torch.tensor([], device=device)
            for images_batch, labels_batch in test_dataset_loader:
                m_inf = model.create_m(images_batch)
                mu0_inf = m_inf.to(torch.float32)

                mu_inf = model.mf_relax(
                    mu0_inf,
                    steps=MF_INF_STEPS,
                    beta=MF_BETA,
                    alpha=MF_DAMP,
                    clamp_nodes=model.input_nodes,
                    clamp_values=m_inf,
                )
                preds = predict_labels_from_mu(mu_inf, model)

                logits = torch.where(preds == labels_batch, 1.0, 0.0)
                acc = torch.cat([acc, logits])

            print(f"epoch {epoch} test result acc:{acc.mean().item()}")
            test_acc.append(acc.mean().item())
            torch.save(model, "model.pth")
            np.savetxt("test_acc.txt", test_acc)

if __name__ == "__main__":
    main()
