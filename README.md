# [Dataflowr：深度学习 DIY](https://dataflowr.github.io/website/)

[![Dataflowr](https://raw.githubusercontent.com/dataflowr/website/master/_assets/dataflowr_logo.png)](https://dataflowr.github.io/website/)

本仓库是深度学习课程 [dataflowr](https://dataflowr.github.io/website/) 的代码与 notebook。以下是 2023 年在巴黎综合理工学院（école polytechnique）的课程安排：

## :sunflower:第一节课：微调 VGG

>- [模块 1 - 引言与总体概览](https://dataflowr.github.io/website/modules/1-intro-general-overview/)
>幻灯片 + 用 VGG 做猫狗分类的 notebook + 实操（更多的猫狗分类）
<details>
  <summary>需要记住的内容</summary>

> - 运行一个深度学习模型并不需要理解所有细节！但本课程的主要目标，就是回过头来把今天做的每一步都弄明白……
> - 要使用 PyTorch 的 dataloader，你需要遵循它的 API（比如做分类任务时，把数据集按类别存放在文件夹里）。
> - 使用预训练模型并稍加改造来适配相似任务，是很容易的。
> - 如果你现在还不明白为什么要用这种损失函数，没关系，模块 3 会讲到。
> - 即使有 GPU，也要避免不必要的计算！

</details>

## :sunflower:第二节课：PyTorch 张量与自动微分

>- [模块 2a - PyTorch 张量](https://dataflowr.github.io/website/modules/2a-pytorch-tensors/)
>- [模块 2b - 自动微分](https://dataflowr.github.io/website/modules/2b-automatic-differentiation/) + 实操
>- 从零实现 MLP：[HW1](https://dataflowr.github.io/website/homework/1-mlp-from-scratch/) 开始
>- [用对偶数和 Julia 再看一眼自动微分](https://github.com/dataflowr/notebooks/blob/master/Module2/AD_with_dual_numbers_Julia.ipynb)
<details>
  <summary>需要记住的内容</summary>

>- PyTorch 张量 = GPU 上的 Numpy + 梯度！
>- 在深度学习中，[广播（broadcasting）](https://numpy.org/doc/stable/user/basics.broadcasting.html) 无处不在，规则与 Numpy 相同。
>- 自动微分不只是链式法则！反向传播算法（或对偶数）是实现自动微分的一个巧妙算法……

 </details>

## :sunflower:第三节课
> - [模块 3 - 分类的损失函数](https://dataflowr.github.io/website/modules/3-loss-functions-for-classification/)
> - [模块 4 - 深度学习中的优化](https://dataflowr.github.io/website/modules/4-optimization-for-deep-learning/)
> - [模块 5 - 堆叠层](https://dataflowr.github.io/website/modules/5-stacking-layers/) 以及在 CIFAR10 上过拟合一个 MLP：[Stacking_layers_MLP_CIFAR10.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module5/Stacking_layers_MLP_CIFAR10.ipynb)
> - [模块 6 - 卷积神经网络](https://dataflowr.github.io/website/modules/6-convolutional-neural-network/)
> - 如何用 dropout 做正则化、用 MC Dropout 做不确定性估计：[模块 15 - Dropout](https://dataflowr.github.io/website/modules/15-dropout/)
<details>
  <summary>需要记住的内容</summary>

>- 损失 vs 准确率。做分类任务时要知道自己的损失函数！
>- 了解你的优化器（模块 4）
>- 知道怎么用 torch.nn.Module 搭神经网络（模块 5）
>- 知道怎么用卷积层和池化层（kernel、stride、padding）
>- 知道怎么用 dropout

</details>

## :sunflower:第四节课
> - [模块 7 - 数据加载](https://dataflowr.github.io/website/modules/7-dataloading/)
> - [模块 8a - Embedding 层](https://dataflowr.github.io/website/modules/8a-embedding-layers/)
> - [模块 8b - 协同过滤](https://dataflowr.github.io/website/modules/8b-collaborative-filtering/) 以及搭建你自己的推荐系统：[08_collaborative_filtering_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module8/08_collaborative_filtering_empty.ipynb)（更大的数据集见 [08_collaborative_filtering_1M.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module8/08_collaborative_filtering_1M.ipynb)）
> - [模块 8c - Word2vec](https://dataflowr.github.io/website/modules/8c-word2vec/) 以及构建你自己的词向量 [08_Word2vec_pytorch_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module8/08_Word2vec_pytorch_empty.ipynb)
> - [模块 16 - BatchNorm](https://dataflowr.github.io/website/modules/16-batchnorm/) 并用 [16_simple_batchnorm_eval.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module16/16_simple_batchnorm_eval.ipynb) 检验你的理解，更多内容见 [16_batchnorm_simple.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module16/16_batchnorm_simple.ipynb)
> - [模块 17 - ResNet](https://dataflowr.github.io/website/modules/17-resnets/) 以及用 [ODIN_mobilenet_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module17/ODIN_mobilenet_empty.ipynb) 把你的分类器改造成分布外检测器
> - [作业 2：类激活图与对抗样本](https://dataflowr.github.io/website/homework/2-CAM-adversarial/) 开始

<details>
  <summary>需要记住的内容</summary>

> - 知道怎么用 dataloader
> - 深度学习中处理类别变量，用 embedding
> - 词向量是从无监督设定出发，构造出一个监督任务（比如预测窗口内的中心词 / 上下文词），再借助负采样学出表示
> - 了解你的 batchnorm
> - 带跳跃连接的架构可以训练更深的模型

</details>

## :sunflower:第五节课
> - [模块 9a：自编码器](https://dataflowr.github.io/website/modules/9a-autoencoders/) 并实现你的带噪自编码器 [09_AE_NoisyAE.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module9/09_AE_NoisyAE.ipynb)
> - [模块 10：生成对抗网络](https://dataflowr.github.io/website/modules/10-generative-adversarial-networks/) 并实现你的 GAN、条件 GAN 和 InfoGAN [10_GAN_double_moon.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module10/10_GAN_double_moon.ipynb)
> - [模块 13：孪生网络与表示学习](https://dataflowr.github.io/website/modules/13-siamese/)
> - [作业 3：用于 MNIST 聚类与生成的 VAE](https://dataflowr.github.io/website/homework/3-VAE/) 开始

## :sunflower:第六节课
> - [模块 11a - 循环神经网络理论](https://dataflowr.github.io/website/modules/11a-recurrent-neural-networks-theory/)
> - [模块 11b - 循环神经网络实践](https://dataflowr.github.io/website/modules/11b-recurrent-neural-networks-practice/) 并用 [11\_predictions\_RNN\_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module11/11_predictions_RNN_empty.ipynb) 预测发动机故障
> - [模块 11c - PyTorch 中带序列的批处理](https://dataflowr.github.io/website/modules/11c-batches-with-sequences/)

## :sunflower:第七节课
> - [模块 12 - 注意力与 Transformer](https://dataflowr.github.io/website/modules/12-attention/)
> - 修正 PyTorch 官方关于 seq2seq 注意力的教程：[12_seq2seq_attention.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module12/12_seq2seq_attention.ipynb)
> - 搭建你自己的 microGPT：[GPT_hist.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module12/GPT_hist.ipynb)
## :sunflower:第八节课
> - [模块 9b - UNet](https://dataflowr.github.io/website/modules/9b-unet/)
> - [模块 9c - 流模型](https://dataflowr.github.io/website/modules/9c-flows/)
> - 实现你自己的 Real NVP：[Normalizing_flows_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module9/Normalizing_flows_empty.ipynb)
## :sunflower:第九节课
> - [模块 18a - 去噪扩散概率模型](https://dataflowr.github.io/website/modules/18a-diffusion/)
> - 在 MNIST 上训练你自己的 DDPM：[ddpm_nano_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module18/ddpm_nano_empty.ipynb)
> - 在 CIFAR10 上微调：[ddpm_micro_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module18/ddpm_micro_sol.ipynb)

更多更新：[![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/marc_lelarge.svg?style=social&label=Follow%20%40marc_lelarge)](https://twitter.com/marc_lelarge)
# :sunflower: 全部 notebook

- [**模块 1：引言与总体概览**](https://dataflowr.github.io/website/modules/1-intro-general-overview/)
    - 引言：微调 VGG 做猫狗分类 [01_intro.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module1/01_intro.ipynb)
    - 实操：用 CNN 做更多猫狗分类 [01_practical_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module1/01_practical_empty.ipynb) 及其解答 [01_practical_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module1/sol/01_practical_sol.ipynb)
- [**模块 2：PyTorch 张量与自动微分**](https://dataflowr.github.io/website/modules/2a-pytorch-tensors/)
    - PyTorch 张量与自动微分基础 [02a_basics.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/02a_basics.ipynb)
    - 从 Numpy 到 PyTorch 的线性回归 [02b_linear_reg.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/02b_linear_reg.ipynb)
    - 实操：从零实现反向传播 [02_backprop.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/02_backprop.ipynb) 及其解答 [02_backprop_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/sol/02_backprop_sol.ipynb)
    - 附加：JAX 入门——函数式自动微分 [autodiff_functional_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/autodiff_functional_empty.ipynb) 及其解答 [autodiff_functional_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/autodiff_functional_sol.ipynb)
    - 附加：JAX 中的线性回归 [linear_regression_jax.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/linear_regression_jax.ipynb)
    - 附加：用对偶数做自动微分 [AD_with_dual_numbers_Julia.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/AD_with_dual_numbers_Julia.ipynb)
- [**作业 1：从零实现 MLP**](https://dataflowr.github.io/website/homework/1-mlp-from-scratch/)
    - [hw1_mlp.ipynb](https://github.com/dataflowr/notebooks/blob/master/HW1/hw1_mlp.ipynb) 及其解答 [hw1_mlp_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/HW1/sol/hw1_mlp_sol.ipynb)
- [**模块 3：分类的损失函数**](https://dataflowr.github.io/website/modules/3-loss-functions-for-classification/)
    - 用多项式回归解释欠拟合与过拟合 [03_polynomial_regression.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module3/03_polynomial_regression.ipynb)
- [**模块 4：深度学习中的优化**](https://dataflowr.github.io/website/modules/4-optimization-for-deep-learning/)
    - 实操：实现 Adagrad、RMSProp、Adam、AMSGrad [04_gradient_descent_optimization_algorithms_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module4/04_gradient_descent_optimization_algorithms_empty.ipynb) 及其解答 [04_gradient_descent_optimization_algorithms_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module4/sol/04_gradient_descent_optimization_algorithms_sol.ipynb)
- [**模块 5：堆叠层**](https://dataflowr.github.io/website/modules/5-stacking-layers/)
    - 实操：在 CIFAR10 上过拟合一个 MLP [Stacking_layers_MLP_CIFAR10.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module5/Stacking_layers_MLP_CIFAR10.ipynb) 及其解答 [MLP_CIFAR10.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module5/sol/MLP_CIFAR10.ipynb)
- [**模块 6：卷积神经网络**](https://dataflowr.github.io/website/modules/6-convolutional-neural-network/)
    - 实操：用 CNN 搭建一个简单的手写数字识别器 [06_convolution_digit_recognizer.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module6/06_convolution_digit_recognizer.ipynb)
- [**作业 2：类激活图与对抗样本**](https://dataflowr.github.io/website/homework/2-CAM-adversarial/)
    - [HW2_CAM_Adversarial.ipynb](https://github.com/dataflowr/notebooks/blob/master/HW2/HW2_CAM_Adversarial.ipynb)

- [**模块 8：Embedding 层**](https://dataflowr.github.io/website/modules/8a-embedding-layers/)、[**协同过滤**](https://dataflowr.github.io/website/modules/8b-collaborative-filtering/) 与 [**Word2vec**](https://dataflowr.github.io/website/modules/8c-word2vec/)
    - 实操：用 Movielens 100k 数据集做协同过滤 [08_collaborative_filtering_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module8/08_collaborative_filtering_empty.ipynb)
    - 实操：重构代码，用 Movielens 1M 数据集做协同过滤 [08_collaborative_filtering_1M.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module8/08_collaborative_filtering_1M.ipynb)
    - 实操：PyTorch 中的词向量（word2vec）[08_Word2vec_pytorch_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module8/08_Word2vec_pytorch_empty.ipynb)
    - 用 GloVe 找同义词和类比 [08_Playing_with_word_embedding.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module8/08_Playing_with_word_embedding.ipynb)
- [**模块 9a：自编码器**](https://dataflowr.github.io/website/modules/9-autoencoders/)
    - 实操：带噪自编码器（含卷积与转置卷积）[09_AE_NoisyAE.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module9/09_AE_NoisyAE.ipynb)
- [**模块 9b - UNet**](https://dataflowr.github.io/website/modules/9b-unet/)
  - 用于图像分割的 UNet [UNet_image_seg.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module9/UNet_image_seg.ipynb)
- [**模块 9c - 流模型**](https://dataflowr.github.io/website/modules/9c-flows/)
  - 实现 Real NVP [Normalizing_flows_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module9/Normalizing_flows_empty.ipynb) 及其解答 [Normalizing_flows_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module9/Normalizing_flows_sol.ipynb)
- [**模块 10 - 生成对抗网络**](https://dataflowr.github.io/website/modules/10-generative-adversarial-networks/)
  - 条件 GAN 与 InfoGAN [10_GAN_double_moon.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module10/10_GAN_double_moon.ipynb)
- [**模块 11 - 循环神经网络**](https://dataflowr.github.io/website/modules/11b-recurrent-neural-networks-practice/) 与 [**PyTorch 中带序列的批处理**](https://dataflowr.github.io/website/modules/11c-batches-with-sequences/)
  - 理论课使用的 notebook：[11_RNN.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module11/11_RNN.ipynb)
  - 用 RNN 预测发动机故障 [11_predictions_RNN_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module11/11_predictions_RNN_empty.ipynb)
- [**模块 12 - 注意力与 Transformer**](https://dataflowr.github.io/website/modules/12-attention/)
  - 修正 [PyTorch 官方教程](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) 中关于 seq2seq 注意力的内容：[12_seq2seq_attention.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module12/12_seq2seq_attention.ipynb) 及其[解答](https://github.com/dataflowr/notebooks/blob/master/Module12/12_seq2seq_attention_solution.ipynb)
  - 搭建一个简单的 transformer block，并像 transformer 一样思考：[GPT_hist.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module12/GPT_hist.ipynb) 及其[解答](https://github.com/dataflowr/notebooks/blob/master/Module12/GPT_hist_sol.ipynb)
- [**模块 13 - 孪生网络与表示学习**](https://dataflowr.github.io/website/modules/13-siamese/)
  - 用对比损失学习 embedding：[13_siamese_triplet_mnist_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module13/13_siamese_triplet_mnist_empty.ipynb)
- [**模块 15 - Dropout**](https://dataflowr.github.io/website/modules/15-dropout/)
  - 在玩具数据集上用 Dropout：[15a_dropout_intro.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module15/15a_dropout_intro.ipynb)
  - 在 MNIST 上玩转 dropout：[15b_dropout_mnist.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module15/15b_dropout_mnist.ipynb)
- [**模块 16 - BatchNorm**](https://dataflowr.github.io/website/modules/16-batchnorm/)
  - batchnorm 的影响：[16_batchnorm_simple.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module16/16_batchnorm_simple.ipynb)
  - 不做任何训练直接玩转 batchnorm：[16_simple_batchnorm_eval.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module16/16_simple_batchnorm_eval.ipynb)
- [**模块 18a - 去噪扩散概率模型**](https://dataflowr.github.io/website/modules/18a-diffusion/)
  - 用于 MNIST 的去噪扩散概率模型：[ddpm_nano_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module18/ddpm_nano_empty.ipynb) 及其解答 [ddpm_nano_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module18/ddpm_nano_sol.ipynb)
  - 用于 CIFAR10 的去噪扩散概率模型：[ddpm_micro_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module18/ddpm_micro_sol.ipynb)
- [**模块 - 图上的深度学习**](https://dataflowr.github.io/website/modules/graph0/)
  - GCN 中的归纳偏置：谱视角 [GCN_inductivebias_spectral.ipynb](https://github.com/dataflowr/notebooks/blob/master/graphs/GCN_inductivebias_spectral.ipynb)，以及 colab 版本 [GCN_inductivebias_spectral-colab.ipynb](https://github.com/dataflowr/notebooks/blob/master/graphs/GCN_inductivebias_spectral-colab.ipynb)
  - PyTorch 中的图卷积网络 [spectral_gnn.ipynb](https://github.com/dataflowr/notebooks/blob/master/graphs/spectral_gnn.ipynb)
-  **NERF**
   -  PyTorch Tiny NERF [tiny_nerf_extended.ipynb](https://github.com/dataflowr/notebooks/blob/master/nerf/tiny_nerf_extended.ipynb)


## 使用方法

如果想在本地运行，请按照[模块 0 - 在本地运行 notebook](https://dataflowr.github.io/website/modules/0-sotfware-installation/) 的说明操作。

## 2020 版课程
历史版本存放在 archive-2020 分支。
