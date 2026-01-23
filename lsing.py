import json
import os
import random
import torch
import numpy as np
import math
import networkx as nx
import scipy.io as sio
from collections import defaultdict
from typing import Optional
# ---- optional line_profiler ----
try:
    from line_profiler import profile  # type: ignore
except Exception:
    def profile(fn):
        return fn

# ---- device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class lsing_model(torch.nn.Module):
    def __init__(
        self,
        input_node_num=28*28,
        label_class_num=10,
        label_node_num=50,
        all_node_num=4264,
        group_init="load",
        group_dir="Data",
    ):
        super().__init__()

        self.input_node_num = input_node_num
        self.label_class_num = label_class_num
        self.label_node_num = label_node_num
        self.all_node_num = all_node_num
        self.input_nodes = None

        self.create_pegasus_pegasus()
        self.group(init_method=group_init, base_dir=group_dir)
        self.create_J_H()

    def create_pegasus_pegasus(self):
        mat = sio.loadmat("Data/JJ_4264.mat")
        W = mat["W"]

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(W.shape[0]))

        if hasattr(W, "tocoo"):
            coo = W.tocoo()
            rows = coo.row
            cols = coo.col
        else:
            rows, cols = np.nonzero(W)

        mask = rows != cols
        edges = zip(rows[mask].tolist(), cols[mask].tolist())
        self.graph.add_edges_from(edges)
        print("create_pegasus_pegasus")

    def group(self, init_method="load", base_dir="Data"):
        colors = np.loadtxt(os.path.join(base_dir, "colorMap_4264.csv"), delimiter=",", dtype=int)
        if colors.ndim != 1:
            colors = colors.reshape(-1)
        node_count = self.graph.number_of_nodes()
        if colors.shape[0] < node_count:
            raise ValueError(
                f"color map length {colors.shape[0]} < node count {node_count}"
            )
        if self.input_node_num + self.label_node_num > node_count:
            raise ValueError("input_node_num + label_node_num exceeds node count")

        self.color_map = {i: int(colors[i]) for i in range(node_count)}
        if init_method == "load":
            self.load_saved_groups(base_dir=base_dir)
        elif init_method == "random":
            self._init_groups_random(node_count, base_dir)
        else:
            raise ValueError(f"unknown init_method: {init_method}")

        self._build_group_all()
        print("grouped")

    def _init_groups_random(self, node_count, base_dir):
        all_nodes = list(range(node_count))
        visible_nodes = set(random.sample(all_nodes, self.input_node_num))
        remaining_nodes = [n for n in all_nodes if n not in visible_nodes]
        label_nodes = set(random.sample(remaining_nodes, self.label_node_num))

        color_map = self.color_map
        self.group_hidden = defaultdict(list)
        self.group_clssify = defaultdict(list)
        self.group_gen = defaultdict(list)
        self.group_label = defaultdict(list)

        for node, color in color_map.items():
            if node in label_nodes:
                self.group_label[color].append(node)

            if node not in visible_nodes:
                self.group_clssify[color].append(node)

            if node not in visible_nodes and node not in label_nodes:
                self.group_hidden[color].append(node)

            # generation: update image bits + hidden bits, but DO NOT update label bits
            if node not in label_nodes:
                self.group_gen[color].append(node)

        self.input_nodes = torch.LongTensor(sorted(visible_nodes)).to(device)
        self._build_label_class_nodes(label_nodes)

        for d in [
            self.group_hidden,
            self.group_clssify,
            self.group_gen,
            self.group_label,
        ]:
            for key, value in d.items():
                value.sort()
                d[key] = torch.LongTensor(value).to(device)
        self._save_group_dict_json(self.group_hidden, os.path.join(base_dir, "group_hidden.json"))
        self._save_group_dict_json(self.group_clssify, os.path.join(base_dir, "group_clssify.json"))
        self._save_group_dict_json(self.group_gen, os.path.join(base_dir, "group_gen.json"))
        self._save_nodes_json(self.input_nodes, os.path.join(base_dir, "group_visible.json"))
        self._save_group_dict_json(self.group_label, os.path.join(base_dir, "group_label.json"))
        self._save_group_dict_json(self.label_class_nodes, os.path.join(base_dir, "label_class_nodes.json"))

    def _build_group_all(self):
        self.group_all = defaultdict(list)
        for node, color in self.color_map.items():
            self.group_all[color].append(node)
        for key, value in self.group_all.items():
            value.sort()
            self.group_all[key] = torch.LongTensor(value).to(device)

    def _build_label_class_nodes(self, label_nodes):
        if self.label_node_num % self.label_class_num != 0:
            raise ValueError("label_node_num must be divisible by label_class_num")

        label_nodes = list(label_nodes)
        if len(label_nodes) != self.label_node_num:
            raise ValueError("label_nodes size does not match label_node_num")

        random.shuffle(label_nodes)
        per_class = self.label_node_num // self.label_class_num
        self.label_class_nodes = {}
        for c in range(self.label_class_num):
            chunk = sorted(label_nodes[c * per_class:(c + 1) * per_class])
            self.label_class_nodes[c] = torch.LongTensor(chunk).to(device)

        self.label_nodes = torch.LongTensor(sorted(label_nodes)).to(device)

    def _build_label_nodes_from_class_nodes(self):
        if not self.label_class_nodes:
            self.label_nodes = None
            return
        ordered = [self.label_class_nodes[k] for k in sorted(self.label_class_nodes.keys())]
        nodes = torch.cat(ordered)
        nodes = torch.unique(nodes)
        nodes, _ = torch.sort(nodes)
        self.label_nodes = nodes

    def _get_visible_nodes(self):
        nodes = []
        if getattr(self, "input_nodes", None) is not None:
            nodes.append(self.input_nodes)
        if getattr(self, "label_nodes", None) is not None:
            nodes.append(self.label_nodes)
        if not nodes:
            return None
        merged = torch.cat(nodes)
        merged = torch.unique(merged)
        merged, _ = torch.sort(merged)
        return merged

    def create_J_H(self):
        self.J = torch.zeros((self.all_node_num, self.all_node_num), device=device)
        for x, y in self.graph.edges:
            x = int(x)
            y = int(y)
            self.J[x, y] = self.J[y, x] = torch.randn(1, device=device) * 0.01

        self.J_mask = torch.where(self.J != 0, 1.0, 0.0).to(device)

        self.H = torch.zeros(self.all_node_num, device=device)
        visible_nodes = self._get_visible_nodes()
        if visible_nodes is None:
            visible_num = self.input_node_num + self.label_node_num
            visible_nodes = torch.arange(visible_num, device=device)
        else:
            visible_num = int(visible_nodes.numel())
        # 这里还是你的初始化方式（训练时你若已改成基于数据pi，也可以保持你自己的）
        bias = math.log((visible_num / self.all_node_num) / (1 - visible_num / self.all_node_num))
        self.H[visible_nodes] = bias

        self.deta_J_all = 0
        self.deta_H_all = 0

    def create_m(self, images_batch=None, labels_batch=None, batch_size=None):
        """
        支持三种用法：
        1) create_m(images_batch, labels_batch)  -> 训练/正相位
        2) create_m(images_batch)               -> 推理/分类
        3) create_m(batch_size=K)               -> 生成：从随机状态开始（无图像输入）
        """
        if images_batch is not None:
            bs = images_batch.shape[0]
        elif labels_batch is not None:
            bs = labels_batch.shape[0]
        elif batch_size is not None:
            bs = int(batch_size)
        else:
            raise ValueError("create_m requires images_batch or labels_batch or batch_size")

        m = torch.randint(0, 2, (bs, self.all_node_num), device=device, dtype=torch.float32)

        # clamp image bits if provided (binary 0/1 expected)
        if images_batch is not None:
            input_nodes = getattr(self, "input_nodes", None)
            if input_nodes is None:
                raise ValueError("input_nodes is None")
            if images_batch.shape[1] != input_nodes.shape[0]:
                raise ValueError("images_batch size does not match input_nodes size")
            m[:, input_nodes] = images_batch.to(m.dtype)

        # clamp label bits if provided
        if labels_batch is not None:
            if getattr(self, "label_class_nodes", None):
                if getattr(self, "label_nodes", None) is None:
                    raise ValueError("label_nodes is None")
                m[:, self.label_nodes] = 0
                for i, label in enumerate(labels_batch):
                    class_nodes = self.label_class_nodes[int(label)]
                    m[i, class_nodes] = 1
            else:
                raise ValueError("label_class_nodes is None")

        # to bipolar {-1,+1}
        m = torch.where(m == 0, -1.0, 1.0)
        return m

    @profile
    def construct(self, m, group, sample_num=1e3, beta=1.0):
        """
        Gibbs updates over a colored node-group.
        增加 beta：用 tanh(beta * I) 做退火。
        """
        # cache sparse rows for each color block (lightweight)
        J_group = [self.J[idxs].to_sparse_coo() for idxs in group.values()]
        H_group = [self.H[idxs] for idxs in group.values()]

        beta = float(beta)

        for _ in range(int(sample_num)):
            for idxs, J, H in zip(group.values(), J_group, H_group):
                I = torch.sparse.mm(J, m.T).T + H
                # annealed update:
                # m_i = sgn( tanh(beta*I_i) - u ), u~U[-1,1]
                u = torch.rand_like(I) * 2 - 1
                new_state = torch.sign(torch.tanh(I * beta) - u)
                # torch.sign can output 0 if exactly equal; map 0 -> +1 to keep bipolar
                new_state = torch.where(new_state == 0, torch.ones_like(new_state), new_state)
                m[:, idxs] = new_state

        return m

    def updateParams(self, m_data, m_model, batch_size, lr=3e-3, momentum=0.6):
        deta_J_new = (torch.mm(m_data.T, m_data) - torch.mm(m_model.T, m_model)) / batch_size
        deta_H_new = (m_data - m_model).mean(axis=0)

        self.deta_J_all = momentum * self.deta_J_all + deta_J_new * lr
        self.deta_H_all = momentum * self.deta_H_all + deta_H_new * lr

        self.J = torch.add(self.J, self.deta_J_all * self.J_mask)
        self.H = torch.add(self.H, self.deta_H_all)


    def save_graph_image(self, path="graph.png", layout="sfdp", max_nodes=None):
        """
        Save a graph visualization using Graphviz layout.
        layout: dot, neato, fdp, sfdp, twopi, circo
        max_nodes: optional cap for a clearer subgraph
        """
        import matplotlib.pyplot as plt

        try:
            from networkx.drawing.nx_agraph import graphviz_layout
        except Exception:
            try:
                from networkx.drawing.nx_pydot import graphviz_layout
            except Exception as exc:
                raise RuntimeError(
                    "Graphviz layout needs pygraphviz or pydot plus Graphviz installed."
                ) from exc

        g = self.graph
        if max_nodes is not None:
            n = min(int(max_nodes), g.number_of_nodes())
            nodes = random.sample(list(g.nodes()), k=n)
            g = g.subgraph(nodes)

        node_color = None
        if hasattr(self, "color_map"):
            palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
            unique = sorted({self.color_map[n] for n in g.nodes()})
            color_lookup = {c: palette[i % len(palette)] for i, c in enumerate(unique)}
            node_color = [color_lookup[self.color_map[n]] for n in g.nodes()]

        pos = graphviz_layout(g, prog=layout)
        plt.figure(figsize=(10, 10))
        nx.draw(g, pos=pos, node_size=6, width=0.2, alpha=0.7, node_color=node_color)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

    def _save_group_dict_json(self, group, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {str(k): v.detach().cpu().tolist() for k, v in group.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=True)

    def _save_nodes_json(self, nodes, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = [int(n) for n in nodes]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=False)

    def _load_group_dict_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): torch.LongTensor(v).to(device) for k, v in data.items()}

    def _load_nodes_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            nodes = []
            for value in data.values():
                nodes.extend(value)
        elif isinstance(data, list):
            nodes = data
        else:
            raise ValueError("invalid nodes json format")
        nodes = sorted({int(n) for n in nodes})
        return torch.LongTensor(nodes).to(device)

    def load_saved_groups(self, base_dir="Data"):
        self.group_hidden = self._load_group_dict_json(os.path.join(base_dir, "group_hidden.json"))
        self.group_clssify = self._load_group_dict_json(os.path.join(base_dir, "group_clssify.json"))
        self.group_gen = self._load_group_dict_json(os.path.join(base_dir, "group_gen.json"))
        self.group_label = self._load_group_dict_json(os.path.join(base_dir, "group_label.json"))
        group_visible_path = os.path.join(base_dir, "group_visible.json")
        if os.path.exists(group_visible_path):
            self.input_nodes = self._load_nodes_json(group_visible_path)
            if int(self.input_nodes.numel()) != self.input_node_num:
                raise ValueError("input_nodes size does not match input_node_num")
        else:
            self.input_nodes = torch.arange(self.input_node_num, device=device)
        label_class_path = os.path.join(base_dir, "label_class_nodes.json")
        if os.path.exists(label_class_path):
            self.label_class_nodes = self._load_group_dict_json(label_class_path)
            self._build_label_nodes_from_class_nodes()
        else:
            self.label_class_nodes = None
            self.label_nodes = None

    @torch.no_grad()
    def nmfa_warm_start(
        self,
        m: torch.Tensor,
        steps: int = 50,
        T_start: float = 5.0,
        T_end: float = 0.5,
        sigma: float = 0.2,
        alpha: float = 0.9,
        clamp_nodes: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
        return_continuous: bool = False,   # NEW
    ) -> torch.Tensor:
        s = m.to(dtype=torch.float32).clone()
        J_sparse = self.J.to_sparse_coo()

        if steps <= 1:
            Ts = [T_end]
        else:
            ratio = (T_end / T_start) ** (1.0 / (steps - 1))
            Ts = [T_start * (ratio ** t) for t in range(steps)]

        for T in Ts:
            Phi = torch.sparse.mm(J_sparse, s.T).T + self.H  # [B, N]

            rms = Phi.pow(2).mean(dim=1, keepdim=True).sqrt() + eps
            Phi = Phi / rms

            if sigma > 0:
                Phi = Phi + torch.randn_like(Phi) * sigma

            s_new = torch.tanh(Phi / max(T, eps))
            s = alpha * s + (1.0 - alpha) * s_new

            if clamp_nodes is not None:
                s[:, clamp_nodes] = m[:, clamp_nodes].to(s.dtype)

        if return_continuous:
            return s  # [-1,1] float

        # 原本的 sign 路径保留（你想继续 warm-start + Gibbs 时还能用）
        out = torch.sign(s)
        out = torch.where(out == 0, torch.ones_like(out), out)
        if clamp_nodes is not None:
            out[:, clamp_nodes] = m[:, clamp_nodes]
        return out.to(m.dtype)

    @torch.no_grad()
    def sample_bipolar_from_soft(
            self,
            s: torch.Tensor,
            clamp_nodes: Optional[torch.Tensor] = None,
            clamp_values: Optional[torch.Tensor] = None,
            n_draws: int = 1,
    ) -> torch.Tensor:
        """
		s: [B, N] in [-1,1]
		线性概率：P(+1) = (s+1)/2
		返回：
		  - n_draws==1: [B, N] bipolar
		  - n_draws>1 : [B, n_draws, N] bipolar
		"""
        p = (s + 1.0) * 0.5
        p = torch.clamp(p, 0.0, 1.0)

        if n_draws == 1:
            u = torch.rand_like(p)
            out = torch.where(u < p, torch.ones_like(p), -torch.ones_like(p))
            if clamp_nodes is not None and clamp_values is not None:
                out[:, clamp_nodes] = clamp_values[:, clamp_nodes]
            return out
        else:
            B, N = p.shape
            u = torch.rand((B, n_draws, N), device=p.device, dtype=p.dtype)
            p3 = p[:, None, :].expand(B, n_draws, N)
            out = torch.where(u < p3, torch.ones_like(u), -torch.ones_like(u))
            if clamp_nodes is not None and clamp_values is not None:
                out[:, :, clamp_nodes] = clamp_values[:, None, clamp_nodes]
            return out
