import opt
from encoder import *



class LowOrder(nn.Module):
    def __init__(self, vae1, ae2, gae1, gae2, n_node=None):
        super(LowOrder, self).__init__()

        self.vae1 = vae1  # 使用变分自编码器
        self.ae2 = ae2    # 使用自编码器

        self.gae1 = gae1
        self.gae2 = gae2

        self.a = Parameter(nn.init.constant_(torch.zeros(n_node, opt.args.n_z), 0.5), requires_grad=True)  # Z_vae, Z_igae
        self.alpha = Parameter(torch.zeros(1))  # ZG, ZL

        self.cluster_centers1 = Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z), requires_grad=True)
        self.cluster_centers2 = Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_centers1.data)
        torch.nn.init.xavier_normal_(self.cluster_centers2.data)
        self.q_distribution1 = q_distribution(self.cluster_centers1)
        self.q_distribution2 = q_distribution(self.cluster_centers2)

        self.label_contrastive_module = nn.Sequential(
            nn.Linear(n_node, opt.args.n_clusters),
            nn.Softmax(dim=1)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def emb_fusion(self, adj, z_vae, z_igae):
        z_i = self.a * z_vae + (1 - self.a) * z_igae
        z_l = torch.spmm(adj, z_i)
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)
        z_g = torch.mm(s, z_l)
        z_tilde = self.alpha * z_g + z_l

        return z_tilde

    def forward(self, x1, adj1, x2, adj2, pretrain=False):
        # 使用 VAE 编码器对 x1 进行编码，返回均值和方差
        mu1, log_var1 = self.vae1.encoder(x1)

        # 使用重参数化技巧进行采样
        z_vae1 = self.reparameterize(mu1, log_var1)

        # 使用 AE 编码器对 x2 进行编码
        z_ae2 = self.ae2.encoder(x2)

        # 使用 IGAE 编码器进行编码
        z_igae1, a_igae1 = self.gae1.encoder(x1, adj1)
        z_igae2, a_igae2 = self.gae2.encoder(x2, adj2)

        # 融合特征
        z1 = self.emb_fusion(adj1, z_vae1, z_igae1)
        z2 = self.emb_fusion(adj2, z_ae2, z_igae2)

        z1_tilde = self.label_contrastive_module(z1.T)
        z2_tilde = self.label_contrastive_module(z2.T)

        cons = [z1, z2, z1_tilde, z2_tilde]

        # VAE 解码器解码
        x_hat1 = self.vae1.decoder(z1)  # 使用 VAE 解码器
        x_hat2 = self.ae2.decoder(z2)    # 使用 AE 解码器

        # IGAE 解码器解码
        z_hat1, z_adj_hat1 = self.gae1.decoder(z1, adj1)
        a_hat1 = a_igae1 + z_adj_hat1

        z_hat2, z_adj_hat2 = self.gae2.decoder(z2, adj2)
        a_hat2 = a_igae2 + z_adj_hat2

        if not pretrain:
            # 计算软分配分布 Q
            Q1 = self.q_distribution1(z1, z_vae1, z_igae1)
            Q2 = self.q_distribution2(z2, z_ae2, z_igae2)
        else:
            Q1, Q2 = None, None

        # 返回 VAE 的均值和方差，用于计算 KL 散度
        return x_hat1, z_hat1, a_hat1, x_hat2, z_hat2, a_hat2, Q1, Q2, z1, z2, cons, mu1, log_var1

