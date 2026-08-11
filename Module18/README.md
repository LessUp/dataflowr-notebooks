# 扩散模型

## 课程
- [去噪扩散概率模型](https://dataflowr.github.io/website/modules/18a-diffusion/)

## MNIST

这个 notebook 在 colab 上训练大约需要 20 分钟。

- [ddpm\_nano\_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module18/ddpm_nano_empty.ipynb) 是让你自己编写 DDPM 算法的 notebook（去噪网络提供了一个简单的 UNet），包含训练和采样两部分。对应的解答在这里：[ddpm\_nano\_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module18/ddpm_nano_sol.ipynb)

## CIFAR10

这个 notebook 在 colab 上训练大约需要 20 分钟（所以不要期待高质量的图片！）。不过，在特定类别上微调后，可以看到模型学到了这些类别的特征。

- [ddpm\_micro\_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module18/ddpm_micro_sol.ipynb)
