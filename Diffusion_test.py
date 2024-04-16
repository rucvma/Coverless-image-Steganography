import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# ``extract``函数的作用是从v这一序列中按照索引t取出需要的数，然后reshape到输入数据x的维度
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    # ``torch.gather``的用法建议看https://zhuanlan.zhihu.com/p/352877584的第一条评论
    # 在此处的所有调用实例中，v都是一维，可以看作是索引取值，即等价v[t], t大小为[batch_size, 1]
    out = torch.gather(v, index=t, dim=0).float().to(device)
    # 再把索引到的值reshape到[batch_size, 1, 1, ...], 维度和x_shape相同
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))
# ``GaussianDiffusionSampler``包含了Diffusion Model的后向过程 & 推理过程

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        """
        所有参数含义和``GaussianDiffusionTrainer``（前向过程）一样
        """
        super().__init__()

        self.model = model
        self.T = T

        # 这里获取betas, alphas以及alphas_bar和前向过程一模一样
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # 这一步是方便后面运算，相当于构建alphas_bar{t-1}
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]  # 把alpha_bar的第一个数字换成1,按序后移

        # 根据公式，后向过程中的计算均值需要用到的系数用coeff1和coeff2表示
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        # 根据公式，计算后向过程的方差
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        """
        该函数用于反向过程中，条件概率分布q(x_{t-1}|x_t)的均值
        Args:
             x_t: 迭代至当前步骤的图像
             t: 当前步数
             eps: 模型预测的噪声，也就是z_t
        Returns:
            x_{t-1}的均值，mean = coeff1 * x_t - coeff2 * eps
        """
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        """
        该函数用于反向过程中，计算条件概率分布q(x_{t-1}|x_t)的均值和方差
        Args:
            x_t: 迭代至当前步骤的图像
            t: 当前步数
        Returns:
            xt_prev_mean: 均值
            var: 方差
        """
        # below: only log_variance is used in the KL computations
        # 这一步我略有不解，为什么要把算好的反向过程的方差大部分替换成betas。
        # 我猜测，后向过程方差``posterior_var``的计算过程仅仅是betas乘上一个(1 - alpha_bar_{t-1}) / (1 - alpha_bar_{t}),
        # 由于1 - alpha_bar_{t}这个数值非常趋近于0，分母为0会导致nan，
        # 而整体(1 - alpha_bar_{t-1}) / (1 - alpha_bar_{t})非常趋近于1，所以直接用betas近似后向过程的方差，
        # 但是t = 1 的时候(1 - alpha_bar_{0}) / (1 - alpha_bar_{1})还不是非常趋近于1，所以这个数值要保留，
        # 因此就有拼接``torch.cat([self.posterior_var[1:2], self.betas[1:]])``这一步
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        # 模型前向预测得到eps(也就是z_t)
        eps = self.model(x_t, t)
        # 计算均值
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        # 反向扩散过程，从x_t迭代至x_0
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            # t = [1, 1, ....] * time_step, 长度为batch_size
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            # 计算条件概率分布q(x_{t-1}|x_t)的均值和方差
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            # 最后一步的高斯噪声设为0（我认为不设为0问题也不大，就本实例而言，t=0时的方差已经很小了）
            if time_step > 0:
                #np.random.seed(time_step)
                #noise=np.random.randn(1,3,64,64)
                #noise=torch.tensor(noise,device='cuda:0').float()
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean #+ torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        # ``torch.clip(x_0, -1, 1)``,把x_0的值限制在-1到1之间，超出部分截断
        return torch.clip(x_0, -1, 1)   


class GaussianDiffusionSamplerTest(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        """
        所有参数含义和``GaussianDiffusionTrainer``（前向过程）一样
        """
        super().__init__()

        self.model = model
        self.T = T

        # 这里获取betas, alphas以及alphas_bar和前向过程一模一样
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # 这一步是方便后面运算，相当于构建alphas_bar{t-1}
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]  # 把alpha_bar的第一个数字换成1,按序后移

        # 根据公式，后向过程中的计算均值需要用到的系数用coeff1和coeff2表示
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        # 根据公式，计算后向过程的方差
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        """
        该函数用于反向过程中，条件概率分布q(x_{t-1}|x_t)的均值
        Args:
             x_t: 迭代至当前步骤的图像
             t: 当前步数
             eps: 模型预测的噪声，也就是z_t
        Returns:
            x_{t-1}的均值，mean = coeff1 * x_t - coeff2 * eps
        """
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        """
        该函数用于反向过程中，计算条件概率分布q(x_{t-1}|x_t)的均值和方差
        Args:
            x_t: 迭代至当前步骤的图像
            t: 当前步数
        Returns:
            xt_prev_mean: 均值
            var: 方差
        """
        # below: only log_variance is used in the KL computations
        # 这一步我略有不解，为什么要把算好的反向过程的方差大部分替换成betas。
        # 我猜测，后向过程方差``posterior_var``的计算过程仅仅是betas乘上一个(1 - alpha_bar_{t-1}) / (1 - alpha_bar_{t}),
        # 由于1 - alpha_bar_{t}这个数值非常趋近于0，分母为0会导致nan，
        # 而整体(1 - alpha_bar_{t-1}) / (1 - alpha_bar_{t})非常趋近于1，所以直接用betas近似后向过程的方差，
        # 但是t = 1 的时候(1 - alpha_bar_{0}) / (1 - alpha_bar_{1})还不是非常趋近于1，所以这个数值要保留，
        # 因此就有拼接``torch.cat([self.posterior_var[1:2], self.betas[1:]])``这一步
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        # 模型前向预测得到eps(也就是z_t)
        eps = self.model(x_t, t)
        # 计算均值
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        # 反向扩散过程，从x_t迭代至x_0
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            # t = [1, 1, ....] * time_step, 长度为batch_size
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            # 计算条件概率分布q(x_{t-1}|x_t)的均值和方差
            np.random.seed(time_step)
            eps=np.random.randn(1,3,64,64)
            eps=torch.tensor(eps,device='cuda:0')
            beta_t=extract(self.betas,t,x_t.shape)
            a=(x_t-beta_t*eps)
            b=(torch.sqrt(1-beta_t))
            x_t=a/b
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        # ``torch.clip(x_0, -1, 1)``,把x_0的值限制在-1到1之间，超出部分截断
        return torch.clip(x_0, -1, 1)   