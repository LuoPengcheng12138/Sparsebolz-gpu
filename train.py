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

from line_profiler import profile

def predict_labels(m, model):
    if getattr(model, "label_class_nodes", None):
        class_keys = sorted(model.label_class_nodes.keys())
        class_nodes = [model.label_class_nodes[k] for k in class_keys]
        scores = torch.stack([m[:, idxs].sum(dim=1) for idxs in class_nodes], dim=1)
        return scores.argmax(dim=1)

    logits = m[:, input_node_num:input_node_num+label_node_num].reshape(
        -1, label_node_num//label_class_num, label_class_num
    )
    return logits.sum(dim=-2).argmax(dim=-1)

@profile
def main():
    model = lsing_model(label_class_num=label_class_num, label_node_num=label_node_num).to(device)

    test_acc = []
    with torch.no_grad():
        for epoch in range(5):
            acc = torch.tensor([]).to(device)
            bar = tqdm(train_dataset_loader)
            for images_batch, labels_batch in bar:
                m = model.create_m(images_batch, labels_batch)
                m_data = model.construct(m, model.group_hidden)

                m = model.create_m(images_batch, labels_batch)
                m_model = model.construct(m, model.group_all)

                model.updateParams(m_data, m_model, batch_size=images_batch.shape[0])

                preds = predict_labels(m, model)
                logits = torch.where(preds==labels_batch, 1., 0.)
                acc = torch.cat([acc, logits])

                bar.set_postfix({
                    "acc" : acc.mean().item()
                })

            acc = torch.tensor([]).to(device)
            for images_batch, labels_batch in test_dataset_loader:
                m = model.create_m(images_batch)
                m = model.construct(m, model.group_clssify)

                preds = predict_labels(m, model)
                logits = torch.where(preds==labels_batch, 1., 0.)
                acc = torch.cat([acc, logits])

            print(f"epoch {epoch} test result acc:{acc.mean().item()}")
            test_acc.append(acc.mean().item())
            torch.save(model, "model.pth")
            np.savetxt("test_acc.txt", test_acc)
if __name__ == "__main__":
    main()
