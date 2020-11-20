<center><b><font size=7>马尔科夫蒙特卡洛详解</font></b></center>

> 本文由 \[简悦 SimpRead\](http://ksria.com/simpread/) 转码， 原文地址 \[zhuanlan.zhihu.com\](https://zhuanlan.zhihu.com/p/250146007)
>
> **参考：**
>
> [MCMC(三)MCMC 采样和 M-H 采样 - 刘建平 Pinard - 博客园​](https://www.cnblogs.com/pinard/p/6638955.html)
>
> [MCMC(四)Gibbs采样 - 刘建平 Pinard - 博客园](https://www.cnblogs.com/pinard/p/6645766.html)

说明：MCMC 主要克服的是高维空间里采样效率过低的问题

> MCMC 算法的一般流程是：先给定目标分布完成采样过程，若目标分布是一维的，就用 M-H 采样方法；若目标分布是多维的，就用 Gibbs 采样方法。采样结束之后，蒙特卡罗方法来用样本集模拟求和，求出目标变量 (期望等) 的统计值作为估计值。这套思路被应用于概率分布的估计、定积分的近似计算、最优化问题的近似求解等问题，特别是被应用于统计学习中概率模型的学习与推理，是重要的统计学习计算方法。

[TOC]

# :pear:蒙特卡罗方法

## 1.1 蒙特卡罗是什么？

> 赌城！  
> 蒙特卡洛是[摩纳哥](https://baike.baidu.com/item/%E6%91%A9%E7%BA%B3%E5%93%A5/127488)公国的一座城市，位于欧洲[地中海](https://baike.baidu.com/item/%E5%9C%B0%E4%B8%AD%E6%B5%B7/11515)之滨、[法国](https://baike.baidu.com/item/%E6%B3%95%E5%9B%BD/1173384)的东南方，属于一个版图很小的国家[摩纳哥公国](https://baike.baidu.com/item/%E6%91%A9%E7%BA%B3%E5%93%A5%E5%85%AC%E5%9B%BD/4428850)，世人称之为 “赌博之国”、“袖珍之国”、“邮票小国”。  
> 蒙特卡洛的赌业，[海洋博物馆](https://baike.baidu.com/item/%E6%B5%B7%E6%B4%8B%E5%8D%9A%E7%89%A9%E9%A6%86/483689)的奇观，[格蕾丝王妃](https://baike.baidu.com/item/%E6%A0%BC%E8%95%BE%E4%B8%9D%E7%8E%8B%E5%A6%83/7059558)的下嫁，都为这个小国增添了许多传奇色  
> 彩，作为世界上人口最密集的一个国度，摩纳哥在仅有 1.95 平方千米的国土上聚集了 3.3 万的人口，可谓地窄人稠。但相对于法国，摩纳哥的地域实在是微乎其微，这个国中之国就像一小滴不慎滴在法国版图内的墨汁，小得不大会引起人去注意它的存在。  
> 蒙特卡罗方法于 20 世纪 40 年代美国在第二次世界大战中研制原子弹的 “曼哈顿计划” 时首先提出，为保密选择用赌城摩纳哥的蒙特卡洛作为代号，因而得名。

看到这里，你可能似乎已经意识到，这个方法一定和**赌博，概率分布，近似数值计算**有着千丝万缕的联系。

事实的确如此，首先我想引用一段李航老师在《统计学习方法》中关于 MCMC 的介绍：

> 蒙特卡罗法 (Monte Carlo method)，也称为统计模拟方法 (statistical simulation  method)，**是通过从概率模型的随机抽样进行近似数值计算的方法**。  马尔可夫链蒙特卡罗法 **(Markov Chain Monte Carlo，MCMC)**，则是以马尔可夫链  (Markovchain) 为**概率模型**的**蒙特卡罗法**。马尔可夫链蒙特卡罗法构建一个马尔可夫链，使其平稳分布就是要进行抽样的分布**，首先基于该马尔可夫链进行随机游走，产生样本的序列，之后使用该平稳分布的样本进行近似的数值计算**。  Metropolis-Hastings 算法是最基本的马尔可夫链蒙特卡罗法，Metropolis 等人在1953 年提出原始的算法，Hastings 在 1970 年对之加以推广，形成了现在的形式。吉布斯抽样(Gibbs sampling) 是更简单、使用更广泛的马尔可夫链蒙特卡罗法，1984年由 S. Geman 和 D. Geman 提出。马尔可夫链蒙特卡罗法被应用于概率分布的估计、定积分的近似计算、最优化问  题的近似求解等问题，特别是被应用于统计学习中概率模型的学习与推理，是重要的统计学习计算方法。

相信读完这一段大多数人都依然懵逼，我们一个概念一个概念地介绍，对于任何一种方法，我们会先阐明它在生活中的用途，让读者有个总体的认识，再去推导它的数学原理，让它彻底地为你所用。

## 1.2 从蒙特卡洛方法说起

**生活中的例子：**

*   **1\. 蒙特卡罗估计圆周率 ![](https://www.zhihu.com/equation?tex=%5Cpi) 的值**

> 这个问题来自小学课本。

绘制一个单位长度的正方形，并绘制四分之一圆。在正方形上随机采样尽可能多的点，放上 3000 次，就可以得到如下这张图：

![](https://pic3.zhimg.com/v2-eb5a7ebd9286d70effbb7bc79ba3bdca_r.jpg)

​    

<center>Fig. 1 蒙特卡罗估计圆周率</center>

洒完这些点以后，圆周率的估计值就很明显了：

![](https://www.zhihu.com/equation?tex=%5Cpi%3D%5Cfrac%7B4N%7D%7B3000%7D)

其中 ![](https://www.zhihu.com/equation?tex=N) 是一个数据点放在圆内部的次数。

*   **2\. 蒙特卡罗估计任意积分的值**

> 这个问题来自中学课本。

![](https://pic2.zhimg.com/v2-51512114814937b87f95f3fb79690fa9_r.jpg)

<center>Fig. 2 特卡罗估计任意积分的值</center>

在该矩形内部，产生大量随机点，可以计算出有多少点落在阴影部分的区域，判断条件为$y_i<f(y_i)$，这个比重就是所要求的积分值，即：

![](https://www.zhihu.com/equation?tex=%5Cfrac%7BN_%7B%E9%98%B4%E5%BD%B1%7D%7D%7BN_%7Btotal%7D%7D%3D%5Cfrac%7BS_%7B%E7%A7%AF%E5%88%86%7D%7D%7BS_%7B%E7%9F%A9%E5%BD%A2%7D%7D)

*   **3\. 蒙特卡罗求解三门问题**

> 这个问题来自中学课本。

三门问题 (Monty Hall problem) 大致出自美国的电视游戏节目 Let's Make a Deal。问题名字来自该节目的主持人蒙提 · 霍尔(Monty Hall)。参赛者会看见三扇关闭了的门，其中一扇的后面有一辆汽车，选中后面有车的那扇门可赢得该汽车，另外两扇门后面则各藏有一只山羊。当参赛者选定了一扇门，但未去开启它的时候，节目主持人开启剩下两扇门的其中一扇，露出其中一只山羊。主持人其后会问参赛者要不要换另一扇仍然关上的门。问题是：换另一扇门会否增加参赛者赢得汽车的机率吗？如果严格按照上述的条件，即主持人清楚地知道，自己打开的那扇门后是羊，那么答案是会。不换门的话，赢得汽车的几率是 1/3。换门的话，赢得汽车的几率是 2/3。

在三门问题中，用 0、1、2 分别代表三扇门的编号，在 \[0,2\] 之间随机生成一个整数代表奖品所在门的编号 prize，再次在 \[0,2\] 之间随机生成一个整数代表参赛者所选择的门的编号 guess。用变量 change 代表游戏中的换门 (ture) 与不换门(false)：

![](https://pic3.zhimg.com/v2-37f27ea78ecf829ab885a324f8d5b58a_r.jpg)

<center>Fig. 3 蒙特卡罗求解三门问题</center>

这样大量重复模拟这个过程（10000 次或者 100000 次）就可以得到换门猜中的概率和不换门猜中的概率。

使用 python 编程实现 (程序太简单就省了，节约版面)，结果为：

```
玩1000000次,每一次都换门：
中奖率为：
0.667383


玩1000000次,每一次都不换门：
中奖率为：
0.333867
```

发现了吗？蒙特卡罗方法告诉我们，换门的中奖率竟是不换门的 2 倍。Why？

![](https://pic1.zhimg.com/v2-50bc8dacfcbff82b88bb887529b8a288_r.jpg)

<center>Fig. 4 蒙特卡罗求解三门问题理解</center>

下面这个例子就能够让你理解这个问题：

比如说主持人和你做游戏，你有一个箱子，里面有 1 个球；主持人一个箱子，里面有 2 个球。他知道每个球的颜色，但你啥也不知道。但是 3 个球里面只有 1 个紫色的球，2 个蓝色的球，谁手里面有紫色的球，谁就获得大奖。

> 主持人说：你要和我换箱子吗？

当然换，我箱子里只有 1 个球，中奖率 ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B3%7D) ，他箱子里有 2 个球，中奖率 ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B2%7D%7B3%7D) ，换的中奖率是不换的 2 倍。

这是这个游戏的结论。现在情况变了：

> 主持人从他的箱子里扔了一个蓝色的球之后说：你要和我换箱子吗？

当然换，我箱子里只有 1 个球，中奖率 ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B3%7D) ，他箱子里有 2 个球，中奖率 ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B2%7D%7B3%7D) 。扔了一个蓝色的，中奖率没变还是 ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B2%7D%7B3%7D) ，换的中奖率是不换的 2 倍。

这是这个游戏的结论。现在情况又变了：

> 没有箱子了，3 个球之前摆了 3 扇门，只有一扇门后面是紫色的球，你只有 1 扇门，主持人有 2 扇，现在，主持人排除了一扇后面是蓝色球的门，再问你：你要和我换门吗？

这种情况和上一种一模一样，只不过去掉了箱子的概念，换成了门而已，当然这不是重要的，你也可以换成铁门，木门，等等。

所以还是换，我只有 1 个门，中奖率 ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B3%7D) ，他有 2 个门，中奖率 ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B2%7D%7B3%7D) 。扔了一个没用的门，中奖率没变还是 ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B2%7D%7B3%7D) ，换的中奖率是不换的 2 倍。

Over！有点跑偏了，怎么说到三门问题上去了？？？本文想表达的只是蒙特卡罗方法可以帮助我们求解三门问题。

*   **4\. 蒙特卡罗估计净利润**

> 这个问题来自小学课本。

证券市场有时交易活跃，有时交易冷清。下面是你对市场的预测。

*   如果交易 Slow，你会以平均价 11 元，卖出 5 万股。
*   如果交易 Hot，你会以平均价 8 元，卖出 10 万股。
*   如果交易 Ok，你会以平均价 10 元，卖出 7.5 万股。
*   固定成本 12 万。

已知你的成本在每股 5.5 元到 7.5 元之间，平均是 6.5 元。请问接下来的交易，你的净利润会是多少？

取 1000 个随机样本，每个样本有两个数值：一个是证券的成本 (5.5 元到 7.5 元之间的均匀分布)，另一个是当前市场状态 (Slow、Hot、Ok，各有 ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B3%7D) 可能)。

![](https://pic2.zhimg.com/v2-10f4726690de715731d672b078677e45_r.jpg)

<center>Fig. 5 蒙特卡罗估计净利润</center>

模拟计算得到，平均净利润为 92, 427 美元。

计算方法是：1000 次抽样，每次按均匀分布随机选定一个成本值、同时按 1/3 的概率选中一个市场情境，然后拿这两个参数可计算出净利润、平均利润，千次加总并取均值就是结果。

![](https://www.zhihu.com/equation?tex=%284.5%2A5%2B1.5%2A10%2B3.5%2A7.5%29%2F3-12+%3D+9.25%28%E4%B8%87%29)

> **问：这些例子的共同点是什么？**

**答：**难度都不超过中学课本 (~~~)。最重要的是，它们都是通过**大量随机样本**，去**了解一个系统**，进而**得到所要计算的值**。

正是由于它的这种特性，所以被称为**统计模拟方法 (statistical simulation method)**，是通过从概率模型的随机抽样进行近似数值计算的方法。

其实随机算法分为两类：蒙特卡罗方法和拉斯维加斯方法，蒙特卡罗方法指的是算法的时间复杂度固定，然而结果有一定几率失败，采样越多结果越好。拉斯维加斯方法指的是算法一定成功，然而运行时间是概率的。

>  **问：你有没有总结出蒙特卡罗方法的使用场景？**

**答：有。**当所求解的问题是**某种随机事件出现的概率**，或者是**某个随机变量的期望值**时，通过某种 "实验"(或者说 "计算机实验" 的方法)，以事件出现的频率作为随机事件的概率 (落在圆内的概率等)，或者得到这个随机变量的某些数字特征 (积分值，净利润等)，并将其作为问题的解。

> 估计概率和估计期望主要的统计学原理为大数定律；

**你也许忽然间明白了，蒙特卡罗方法应该这么用：**

比如说我要求某个**参量**，直接求解遇到了困难，那我就构造一个合适的**概率模型**，对这个模型进行大量的采样和统计实验，使它的**某些统计参量**正好是**待求问题的解**，那么，只需要把这个参量的值统计出来，那么问题的解就得到了估计值。

## 1.3 随机抽样

### 1.3.1 介绍

通过上面的几个例子我们发现：蒙特卡罗法要解决的问题是：**假设概率分布的定义已知**，通过抽样获得概率分布的随机样本，并**通过得到的随机样本对概率分布的特征进行分析**。比如，从样本得到经验分布，从而估计总体分布；或者从样本计算出样本均值，从而估计总体期望。所以蒙特卡罗法的核心是**随机抽样 (random sampling)**。

可是，随机抽样 (random sampling) 的方法从来都不是一成不变的，在下面的这个例子里面，我会阐明马尔科夫方法的一般形式：

求解积分： ![](https://www.zhihu.com/equation?tex=S%3D%5Cint_%7Ba%7D%5E%7Bb%7Df%28x%29dx)

![](https://pic4.zhimg.com/v2-62f1028521159e0fc83930cc1561632f_r.jpg)

如果很难求出 ![](https://www.zhihu.com/equation?tex=f%28x%29) 的具体表达式，那么我们就需要用到蒙特卡洛方法。你可以如前文所述在二维空间中洒 1000 个点，看有多少落在积分区域的内部。也可以换种方式理解这个做法：

![](https://www.zhihu.com/equation?tex=S%3D%5Cint_%7Ba%7D%5E%7Bb%7Df%28x%29dx%3D%5Cint_%7Ba%7D%5E%7Bb%7Dp%28x%29%5Cfrac%7Bf%28x%29%7D%7Bp%28x%29%7Ddx%3DE_%7Bx%5Csim+p%28x%29%7D%5B%5Cfrac%7Bf%28x%29%7D%7Bp%28x%29%7D%5D)

注意我们对这个积分表达式进行了一些 trick，使它变成了一个对 ![](https://www.zhihu.com/equation?tex=%5Cfrac%7Bf%28x%29%7D%7Bp%28x%29%7D) 的期望，这个期望服从的分布是 ![](https://www.zhihu.com/equation?tex=p%28x%29) 。注意这个分布 ![](https://www.zhihu.com/equation?tex=p%28x%29)可以是任何一种概率分布。

下面如果想估计这个期望值，就需要按照概率分布 ![](https://www.zhihu.com/equation?tex=p%28x%29+)独立地抽取 n 个样本 ![](https://www.zhihu.com/equation?tex=x_%7B1%7D%2Cx_%7B2%7D%2C...%2Cx_%7Bn%7D) 。

注意，这里不是胡乱抽取 ![](https://www.zhihu.com/equation?tex=n) 个样本，而是按照概率分布 ![](https://www.zhihu.com/equation?tex=p%28x%29+)独立地抽取 ![](https://www.zhihu.com/equation?tex=n) 个样本。 ![](https://www.zhihu.com/equation?tex=p%28x%29) 不同，抽取样本的方式当然也不会相同。

大数定律告诉我们，当 ![](https://www.zhihu.com/equation?tex=n%5Crightarrow%2B+%5Cinfty) 时， ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Cfrac%7Bf%28x_%7Bi%7D%29%7D%7Bp%28x_%7Bi%7D%29%7D%7D%5Crightarrow+E_%7Bx%5Csim+p%28x%29%7D%5B%5Cfrac%7Bf%28x%29%7D%7Bp%28x%29%7D%5D)

这句话的意思是说：我按照概率分布 ![](https://www.zhihu.com/equation?tex=p%28x%29+)独立地抽取 ![](https://www.zhihu.com/equation?tex=n) 个样本，只要把 ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Cfrac%7Bf%28x_%7Bi%7D%29%7D%7Bp%28x_%7Bi%7D%29%7D%7D) 算出来， ![](https://www.zhihu.com/equation?tex=+E_%7Bx%5Csim+p%28x%29%7D%5B%5Cfrac%7Bf%28x%29%7D%7Bp%28x%29%7D%5D) 也就得到了。

一个特殊的情况是当按均匀分布抽取 ![](https://www.zhihu.com/equation?tex=n) 个样本时，即 ![](https://www.zhihu.com/equation?tex=p%28x%29%3D%5Cfrac%7B1%7D%7Bb-a%7D)：

![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Cfrac%7Bf%28x_%7Bi%7D%29%7D%7Bp%28x_%7Bi%7D%29%7D%7D%3D%5Cfrac%7Bb-a%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bf%28x_i%29%7D)

这就是大一高数里面讲的积分的估计方法：

> 把积分区域等分为 ![](https://www.zhihu.com/equation?tex=n+) 份 (**均匀抽样**)，每一份取一个点 ![](https://www.zhihu.com/equation?tex=x_%7Bi%7D) ，计算出 ![](https://www.zhihu.com/equation?tex=f%28x_i%29) 取均值作为这一段积分的函数的均值，最后乘以积分区间的长度 ![](https://www.zhihu.com/equation?tex=%28b-a%29) 即可。

也就是蒙特卡罗方法的一个特例而已。

**说了这么半天，它的基本思想就能用一个公式表达：**

![](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Cfrac%7Bf%28x_%7Bi%7D%29%7D%7Bp%28x_%7Bi%7D%29%7D%7D%5Crightarrow+E_%7Bx%5Csim+p%28x%29%7D%5B%5Cfrac%7Bf%28x%29%7D%7Bp%28x%29%7D%5D) ，注意 ![](https://www.zhihu.com/equation?tex=n) 个点**要按照 ![](https://www.zhihu.com/equation?tex=p%28x%29) 采样**，且 ![](https://www.zhihu.com/equation?tex=n) 越大，越精确。

现在的问题转到了如何按照 ![](https://www.zhihu.com/equation?tex=p%28x%29) 采样若干个样本上来。但是，**按照** ![](https://www.zhihu.com/equation?tex=p%28x%29) **采样绝非易事**，因为有些时候我们根本不知道 ![](https://www.zhihu.com/equation?tex=p%28x%29) 是什么，或者有时候是一个很复杂的表达式，计算机没法直接抽样。

### 1.3.2 随机抽样方法3：直接采样

直接采样的思想是，通过对均匀分布采样，实现对任意分布的采样。因为均匀分布采样好猜，我们想要的分布采样不好采，那就采取一定的策略通过简单采取求复杂采样。

假设$y$服从某项分布$p(y)$，其累积分布函数$CDF$为$h(y)$，有样本$z\sim Uniform(0,1)$，我们令 $z = h(y)$，即 $y = h^{-1}(z)$，结果$y$即为对分布$p(y)$的采样。

![img](https://upload-images.jianshu.io/upload_images/314331-a158633099c6fd5c.png?imageMogr2/auto-orient/strip|imageView2/2/w/1165/format/webp)

举个例子：

![img](https://upload-images.jianshu.io/upload_images/314331-fd17a2a603ba3a0d.png?imageMogr2/auto-orient/strip|imageView2/2/w/1123/format/webp)

直接采样有一个问题，就是上图下方的那两个问号。

### 1.3.3 随机抽样方法 2：拒绝 - 接受采样

> **这种方法适用于$p(z)$极度复杂不规则的情况。**

接受-拒绝采样的思想是，对于分布$p(z)$，很难通过直接采样进行采样，但是我们可以找一个可以直接采样分布$q(z)$作为媒介，这个媒介的专业术语为建议分布(proposal distribution)，可以通过直接采样或其他采样进行采样，那么我们怎么可以通过$q(z)$采样得到$p(z)$的采样呢？

先看下图：

![img](https:////upload-images.jianshu.io/upload_images/314331-61e4cda1f39a3454.png?imageMogr2/auto-orient/strip|imageView2/2/w/586/format/webp)

红色的是$p(z)$, 蓝色的是$q(z)$，我们对$q(z)$乘一个参数$k$，让$k$能正好包住$p(z)$，那么对于每一个从$q(z)$得到的样本$z_0$，我们有一定的概率接受它，概率的大小就是$p(z_0) / kq(z_0)$。很容易就能看出来，在$p(z)$和$kq(z)$相切的地方的采样，接受率就是1。

那么有人问了，接受率能计算出来，但是我们对于一个样本$z_0$，到底怎么判断是接受还是不接受啊？我们有$u\sim Uniform[0,1]$，对于每一个样本$z_0$，我们一个$u_0$，如果$u_0 \le p(z_0) / kq(z_0)=\alpha$($\alpha$为接受率)，我们就接受，否则就拒绝。重复此过程，得到的样本就服从分布$p(z)$。

![img](https:////upload-images.jianshu.io/upload_images/314331-1e6550af3180c86c.png?imageMogr2/auto-orient/strip|imageView2/2/w/504/format/webp)

当然，$q(z)$的选取要有一定规则：$q(z)$与$p(z)$外形要相近，$q(z)$采样方便。

现在有一个问题，$k$怎么求？

其实也很简单，看图有 $kq(z) \ge p(z)$，那么$k \ge p(z) / q(z)$，我们求$p(z) / q(z)$的最大值，即为$k$。

举个例子，对截断正态分布的接受-拒绝采样。

截断正态分布的意思就是对于正态分布$N(a, b)$ ，$x$属于[0, 4]，其在[0, 4]上的积分为1，而不是在负无穷到正无穷的积分为1。截断正态分布不是正态分布，所以，我们知道截断正态分布的概率密度函数。

维基上对截断正态分布的定义：



![img](https:////upload-images.jianshu.io/upload_images/314331-8adaceae2df6a6ff.png?imageMogr2/auto-orient/strip|imageView2/2/w/852/format/webp)

分子的小fai是标准正态分布的概率密度函数，分母上的$\Phi$是标准正态分布的累积分布函数。

我们有$p(z)$服从$N(1, 1)$， $I(0 <= x <= 4)$，令$q(z)~U[0, 4]$，根据上图的公式，$p(z)$就有了，$q(z) = 1/4$，所以$k = \max(p(z) / q(z))$，在$z = 1$（均值）的时候$p(z) / q(z)$取最大，所以得到$k$：



![img](https:////upload-images.jianshu.io/upload_images/314331-40444867f581f781.png?imageMogr2/auto-orient/strip|imageView2/2/w/485/format/webp)

这个例子比较巧，分母没有$z$，我们可以直接判断在$z=1$时，$p(z)/q(z)$取最大，如果$p(z)$, $q(z)$都有$z$，那么要通过求导的方式求$k$了。

### 1.3.4 随机抽样方法 3：重采样技术 reparameterization trick

> **这个方法在 VAE 中经常使用，可以参考我之前的 Blog，这种方法适用于 ![](https://www.zhihu.com/equation?tex=p%28x%29) 是常见的连续分布，比如正态分布，t 分布，F 分布，Beta 分布，Gamma 分布等。**

在 VAE 中使用重采样技术是为了能让网络能够完成反向传播，具体是这样子：

现在要从 ![](https://www.zhihu.com/equation?tex=N%28%5Cmu%2C%5Csigma%5E%7B2%7D%29) 中采样一个 ![](https://www.zhihu.com/equation?tex=Z) ，相当于从 ![](https://www.zhihu.com/equation?tex=N%280%2C1%29) 中采样一个 ![](https://www.zhihu.com/equation?tex=%5Cvarepsilon) ，然后让

![](https://www.zhihu.com/equation?tex=Z%3D%5Cmu%2B%5Cvarepsilon%5Ctimes%5Csigma)

于是，我们将从 ![](https://www.zhihu.com/equation?tex=N%28%5Cmu%2C%5Csigma%5E%7B2%7D%29) 采样变成了从 ![](https://www.zhihu.com/equation?tex=N%280%2C1%29) 中采样，然后通过参数变换得到从 ![](https://www.zhihu.com/equation?tex=N%28%5Cmu%2C%5Csigma%5E%7B2%7D%29) 中采样的结果。这样一来，“采样” 这个操作就不用参与梯度下降了，改为采样的结果参与，使得整个模型可训练了。

> **重参数技巧**还可以这样来理解：  
> 比如我有 ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+Z%7D) ，可是 Z 是采样得到的，后向传播无法继续往前面求导了。  
> 现在使用了**重参数技巧**以后， ![](https://www.zhihu.com/equation?tex=Z%3D%5Cmu%2B%5Cvarepsilon%5Ctimes%5Csigma) 。 ![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+%5Csigma%5E%7B2%7D%7D%3D%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+Z%7D%5Cfrac%7B%5Cpartial+Z%7D%7B%5Cpartial+%5Csigma%5E%7B2%7D%7D%3D%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+Z%7D%5Cvarepsilon)  
> 这样就可以正常使用反向传播了。

所以说重采样技术的思想就是把**复杂分布的采样**转化为**简单分布的采样**，再通过一些**变换**把采样结果变回去。

再比如我要从**二维正态分布**中采样得到相互独立的 ![](https://www.zhihu.com/equation?tex=X%2CY%5Csim+N%28%5Cmu_1%2C%5Cmu_2%2C%5Csigma_%7B1%7D%5E%7B2%7D%2C%5Csigma_%7B2%7D%5E%7B2%7D%29%3DN%280%2C0%2C1%2C1%29) ，我就可以先从**均匀分布**中采样两个随机变量 ![](https://www.zhihu.com/equation?tex=U_1%2CU_2%5Csim+U%280%2C1%29) ，再通过下面的变换得到 ![](https://www.zhihu.com/equation?tex=X%2CY) ：

![](https://www.zhihu.com/equation?tex=X%3Dcos%282%5Cpi+U_1%29%5Csqrt%7B-2lnU_2%7D)

![](https://www.zhihu.com/equation?tex=Y%3Dsin%282%5Cpi+U_1%29%5Csqrt%7B-2lnU_2%7D)

这个变换的专业术语叫做 Box-Muller 变换，它的证明如下：

> 证：假设相互独立的 ![](https://www.zhihu.com/equation?tex=X%2CY%5Csim+N%280%2C0%2C1%2C1%29) ，则：  
> ![](https://www.zhihu.com/equation?tex=X) 的概率密度： ![](https://www.zhihu.com/equation?tex=p%28X%29%3D%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D%7De%5E%7B-%5Cfrac%7Bx%5E2%7D%7B2%7D%7D) ， ![](https://www.zhihu.com/equation?tex=Y) 的概率密度： ![](https://www.zhihu.com/equation?tex=p%28Y%29%3D%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D%7De%5E%7B-%5Cfrac%7By%5E2%7D%7B2%7D%7D)  
> 因为相互独立，所以联合概率密度为： ![](https://www.zhihu.com/equation?tex=p%28X%2CY%29%3D%5Cfrac%7B1%7D%7B2%5Cpi%7De%5E%7B-%5Cfrac%7Bx%5E2%2By%5E2%7D%7B2%7D%7D)  
> 使用二重积分的经典套路，将 X、Y 作坐标变换，使：  
> ![](https://www.zhihu.com/equation?tex=X%3DRcos%5Ctheta%5C%5CY%3DRsin%5Ctheta)  
> 得到： ![](https://www.zhihu.com/equation?tex=p%28R%2C%5Ctheta%29%3D%5Cfrac%7B1%7D%7B2%5Cpi%7De%5E%7B-%5Cfrac%7Br%5E2%7D%7B2%7D%7D)  
> 而且： ![](https://www.zhihu.com/equation?tex=%5Cint_%7B-+%5Cinfty%7D%5E%7B%2B+%5Cinfty%7D+%5Cint_%7B-+%5Cinfty%7D%5E%7B%2B+%5Cinfty%7D+%5Cfrac%7B1%7D%7B2%5Cpi%7De%5E%7B-%5Cfrac%7Bx%5E2%2By%5E2%7D%7B2%7D%7DdXdY%3D%5Cint_%7B0%7D%5E%7B2%5Cpi%7D%5Cint_%7B0%7D%5E%7B%2B+%5Cinfty%7D%5Cfrac%7B1%7D%7B2%5Cpi%7De%5E%7B-%5Cfrac%7Br%5E2%7D%7B2%7D%7Drdrd%5Ctheta%3D1)  
> **算到这里要明确我们的目标是什么？**  
> **答：**应该是反推出来随机变量 ![](https://www.zhihu.com/equation?tex=U_1%2CU_2%5Csim+U%280%2C1%29) 。先看看 ![](https://www.zhihu.com/equation?tex=R) 和 ![](https://www.zhihu.com/equation?tex=%5Ctheta) 的概率分布和概率密度吧，根据概率论知识有：  
> ![](https://www.zhihu.com/equation?tex=F_%7BR%7D%28r%29%3D%5Cint_%7B0%7D%5E%7B2%5Cpi%7D%5Cint_%7B0%7D%5E%7Br%7D%5Cfrac%7B1%7D%7B2%5Cpi%7De%5E%7B-%5Cfrac%7Br%5E2%7D%7B2%7D%7Drdrd%5Ctheta%3D1-e%5E%7B-%5Cfrac%7Br%5E2%7D%7B2%7D%7D)  
> ![](https://www.zhihu.com/equation?tex=F_%7B%5Ctheta%7D%28%5Cvarphi%29%3D%5Cint_%7B0%7D%5E%7B%5Cvarphi%7D%5Cint_%7B0%7D%5E%7B%2B+%5Cinfty%7D%5Cfrac%7B1%7D%7B2%5Cpi%7De%5E%7B-%5Cfrac%7Br%5E2%7D%7B2%7D%7Drdrd%5Ctheta%3D%5Cfrac%7B%5Cvarphi%7D%7B2%5Cpi%7D)  
> 一眼就看出来 ![](https://www.zhihu.com/equation?tex=%5Ctheta%5Csim+U%280%2C2%5Cpi%29) 。所以 ![](https://www.zhihu.com/equation?tex=%5Ctheta%3D2%5Cpi+U_1%3D2%5Cpi+U_2) 。  
> 接下来设 ![](https://www.zhihu.com/equation?tex=Z%3D1-e%5E%7B-%5Cfrac%7BR%5E2%7D%7B2%7D%7D) ，不知道 ![](https://www.zhihu.com/equation?tex=Z) 服从什么分布哦。  
> ![](https://www.zhihu.com/equation?tex=P%28Z%3Cz%29%3DP%281-e%5E%7B-%5Cfrac%7BR%5E2%7D%7B2%7D%7D%3Cz%29%3DP%28R%3C%5Csqrt%7B-2ln%281-z%29%7D%29%3DF_%7BR%7D%28%5Csqrt%7B-2ln%281-z%29%7D%29%3Dz)  
> 所以 ![](https://www.zhihu.com/equation?tex=Z%3D1-e%5E%7B-%5Cfrac%7BR%5E2%7D%7B2%7D%7D%5Csim+U%280%2C1%29) ，现在知道了 ![](https://www.zhihu.com/equation?tex=Z) 服从均匀分布。  
> 所以此时有 ![](https://www.zhihu.com/equation?tex=1-e%5E%7B-%5Cfrac%7BR%5E2%7D%7B2%7D%7D%5Csim+U%280%2C1%29) ，即 ![](https://www.zhihu.com/equation?tex=e%5E%7B-%5Cfrac%7BR%5E2%7D%7B2%7D%7D%5Csim+U%280%2C1%29) ，即有：![](https://www.zhihu.com/equation?tex=R%3D%5Csqrt%7B-2lnU_1%7D%3D%5Csqrt%7B-2lnU_2%7D)  
> 所以：  
> ![](https://www.zhihu.com/equation?tex=X%3DRcos%5Ctheta%3Dcos%282%5Cpi+U_1%29%5Csqrt%7B-2lnU_2%7D)  
> ![](https://www.zhihu.com/equation?tex=Y%3DRsin%5Ctheta%3Dsin%282%5Cpi+U_1%29%5Csqrt%7B-2lnU_2%7D)  
> Box-Muller 变换得证。

至此，我们讲完了蒙特卡洛方法，这个方法非常强大和灵活，也很容易实现。对于许多问题来说，它往往是最简单的计算方法，有时甚至是唯一可行的方法。

从上面可以看出，要想将蒙特卡罗方法作为一个通用的采样模拟求和的方法，必须解决如何方便得到各种**复杂概率分布**的对应的**采样样本集**的问题。==而马尔科夫链就能帮助你找到这些**复杂概率分布**的对应的**采样样本集。**==

# :peach: 马尔科夫链

## 2.1 马尔科夫的初步介绍

==说明：==

下文中涉及的转移概率矩阵与我们常用的不太一样，为转置后的概率矩阵，同样的状态分布向量也应该是一个行向量

在蒙特卡罗方法中，我们采集大量的样本，构造一个合适的**概率模型**，对这个模型进行大量的采样和统计实验，使它的**某些统计参量**正好是**待求问题的解。**但是，我们需要大量采样，虽然我们有**拒绝 - 接受采样**和**重采样技术**，但是依旧面临采样困难的问题。巧了，马尔科夫链可以帮我们解决这个难题。

首先我们看一些基本的定义：

> **定义 (马尔可夫链)：**考虑一个随机变量的序列 ![](https://www.zhihu.com/equation?tex=X%3D%5Cleft%5C%7B+X_0%2CX_1%2C...%2CX_t%2C...+%5Cright%5C%7D)  
> 这里 ![](https://www.zhihu.com/equation?tex=X_t) 表示时刻 ![](https://www.zhihu.com/equation?tex=t) 的随机变量， ![](https://www.zhihu.com/equation?tex=t%3D0%2C1%2C2%2C...) 。每个随机变量 ![](https://www.zhihu.com/equation?tex=X_t) 的  
> 取值集合相同，称为状态空间，表示为 ![](https://www.zhihu.com/equation?tex=S) 。随机变量可以是离散的，也可以是连续的。  
> 以上随机变量的序列构成**随机过程 (stochastic process)**。  
> 假设初始时刻的随机变量 ![](https://www.zhihu.com/equation?tex=X_0) 遵循概率分布 ![](https://www.zhihu.com/equation?tex=P%28X_0%29%3D%5Cpi_0) ，称为初始状态分布。在某个时刻 ![](https://www.zhihu.com/equation?tex=t%5Cgeq1) 的随机变量 ![](https://www.zhihu.com/equation?tex=X_t) 与前一个时刻的随机变量 ![](https://www.zhihu.com/equation?tex=X_%7Bt-1%7D) 之间有条件分布 ![](https://www.zhihu.com/equation?tex=P%28X_%7Bt%7D%7CX_%7Bt-1%7D%29) ，如果 ![](https://www.zhihu.com/equation?tex=X_t) 只依赖于 ![](https://www.zhihu.com/equation?tex=X_%7Bt-1%7D) ，而不依赖于过去的随机变量 ![](https://www.zhihu.com/equation?tex=%5Cleft%5C%7B+X_0%2CX_1%2C...%2CX_%7Bt-2%7D+%5Cright+%5C%7D) ，这一性质称为马尔可夫性，即：  
> ![](https://www.zhihu.com/equation?tex=P%28X_%7Bt%7D%7CX_%7Bt-1%7D%2CX_%7Bt-2%7D%2C...%2CX_%7B0%7D%29%3DP%28X_%7Bt%7D%7CX_%7Bt-1%7D%29%2Ct%3D1%2C2%2C...)  
> 具有马尔可夫性的随机序列 ![](https://www.zhihu.com/equation?tex=X%3D%5Cleft%5C%7B+X_0%2CX_1%2C...%2CX_t%2C...+%5Cright%5C%7D) 称为马尔可夫链 (Markov  chain)，或马尔可夫过程 (Markov process)。条件概率分布 ![](https://www.zhihu.com/equation?tex=P%28X_%7Bt%7D%7CX_%7Bt-1%7D%29) 称为马尔可夫链的转移概率分布。转移概率分布决定了马尔可夫链的特性。

如果这个**条件概率分布**与**具体的时刻** ![](https://www.zhihu.com/equation?tex=t) 是无关的，则称这个马尔科夫链为时间齐次的马尔可夫链 (time homogenous Markov chain)。

**定义：**转移概率矩阵：

![](https://www.zhihu.com/equation?tex=P%3D%5Cbegin%7Bbmatrix%7Dp_%7B11%7D+%26+p_%7B12%7D+%26p_%7B13%7D+%5C%5C+p_%7B21%7D+%26+p_%7B22%7D+%26p_%7B23%7D+%5C%5C+p_%7B31%7D+%26+p_%7B32%7D+%26p_%7B33%7D%5Cend%7Bbmatrix%7D)

其中 ![](https://www.zhihu.com/equation?tex=p_%7Bij%7D%3DP%28X_%7Bt%7D%3Di%7CX_%7Bt-1%7D%3Dj%29) 。

定义：马尔科夫链在 ![](https://www.zhihu.com/equation?tex=t) 时刻的概率分布称为 ![](https://www.zhihu.com/equation?tex=t) 时刻的状态分布：

![](https://www.zhihu.com/equation?tex=%5Cpi+%28t%29%3D%5Cbegin%7Bbmatrix%7D%5Cpi_%7B1%7D%28t%29+%5C%5C+%5Cpi_%7B2%7D%28t%29+%5C%5C+%5Cpi_%7B3%7D%28t%29%5Cend%7Bbmatrix%7D)

其中 ![](https://www.zhihu.com/equation?tex=%5Cpi_%7Bi%7D+%28t%29%3DP%28X_%7Bt%7D%3Di%29%2Ci%3D1%2C2%2C...) 。

特别地，马尔可夫链的初始状态分布可以表示为：

![](https://www.zhihu.com/equation?tex=%5Cpi+%280%29%3D%5Cbegin%7Bbmatrix%7D%5Cpi_%7B1%7D%280%29+%5C%5C+%5Cpi_%7B2%7D%280%29+%5C%5C+%5Cpi_%7B3%7D%280%29%5Cend%7Bbmatrix%7D)

通常初始分布 ![](https://www.zhihu.com/equation?tex=%5Cpi+%280%29) 向量只有一个分量是 1，其余分量都是 0，表示马尔可夫链从一个具体状态开始。

有限离散状态的马尔可夫链可以由有向图表示。结点表示状态，边表示状态之间的转移，边上的数值表示转移概率。从一个初始状态出发，根据有向边上定义的概率在状态之间随机跳转 (或随机转移)，就可以产生**状态的序列**。马尔可夫链实际上是刻画**随时间在状态之间转移**的模型，假设未来的转移状态只依赖于现在的状态，而与过去的状态无关。

下面通过一个简单的例子给出马尔可夫链的直观解释：如下图王者荣耀玩家选择英雄的职业的转变，转移概率矩阵为：

![](https://www.zhihu.com/equation?tex=P%3D%5Cbegin%7Bbmatrix%7D0.5+%26+0.5+%260.25+%5C%5C+0.25+%26+0+%260.25+%5C%5C+0.25+%26+0.5+%26+0.5%5Cend%7Bbmatrix%7D)

![](https://pic2.zhimg.com/v2-696651b37f2dc6a472c314f1ba78194d_r.jpg)

<center>Fig. 7 王者荣耀玩家选择英雄的职业的转变</center>

我们试图用程序去模拟这个状态的变化情况，任意假设一个初始状态：设初始的三个概率分别是![](https://www.zhihu.com/equation?tex=%5B0.5%2C0.3%2C0.2%5D)，即![](https://www.zhihu.com/equation?tex=t_%7B0%7D)时刻，50% 概率选择射手，30% 概率选择辅助，20% 概率选择打野，将此代入转移概率，我们一直计算到![](https://www.zhihu.com/equation?tex=t_%7B100%7D)看看是什么情况：

```python
import numpy as np
import matplotlib.pyplot as plt
transfer_matrix = np.array([[0.5,0.5,0.25],[0.25,0,0.25],[0.25,0.5,0.5]],dtype='float32')
start_matrix = np.array([[0.5],[0.3],[0.2]],dtype='float32')

value1 = []
value2 = []
value3 = []
for i in range(30):
    start_matrix = np.dot(transfer_matrix, start_matrix)
    value1.append(start_matrix[0][0])
    value2.append(start_matrix[1][0])
    value3.append(start_matrix[2][0])
print(start_matrix)

x = np.arange(30)
plt.plot(x,value1,label='Archer')
plt.plot(x,value2,label='Support')
plt.plot(x,value3,label='Assassin')
plt.legend()
plt.show()
```

![](https://pic2.zhimg.com/v2-962d1d4742dbed02879d33da6184bfa5_r.jpg)

可以发现，从 5 轮左右开始，我们的状态概率分布就不变了，一直保持在：

```python
[[0.4       ]
 [0.19999999]
 [0.39999998]]
```

从这个实验我们得出了一个结论：这个玩家如果一直把王者荣耀玩下去，他最后选择射手，辅助，打野的概率会趋近于： ![](https://www.zhihu.com/equation?tex=%5B0.4%2C0.2%2C0.4%5D) ，很有意思的结论。

*   **问：是不是所有马尔科夫链都有平稳分布？**

**答：**不一定，必须满足下面的定理：

> **定理：**给定一个马尔科夫链 ![](https://www.zhihu.com/equation?tex=X%3D%5Cleft%5C%7B+X_0%2CX_1%2C...%2CX_t%2C...+%5Cright%5C%7D) ， ![](https://www.zhihu.com/equation?tex=t) 时刻的状态分布：  
> ![](https://www.zhihu.com/equation?tex=%5Cpi%3D%28%5Cpi_1%2C%5Cpi_2%2C...%29) 是 ![](https://www.zhihu.com/equation?tex=X) 的平稳分布的条件是 ![](https://www.zhihu.com/equation?tex=%5Cpi%3D%28%5Cpi_1%2C%5Cpi_2%2C...%29) 是下列方程组的解：  
> ![](https://www.zhihu.com/equation?tex=x_%7Bi%7D%3D%5Csum_%7Bj%7D%7Bp_%7Bij%7Dx_j%7D%2Ci%3D1%2C2%2C...)  
> ![](https://www.zhihu.com/equation?tex=x_i%5Cgeq0%2Ci%3D1%2C2%2C...)  
> ![](https://www.zhihu.com/equation?tex=%5Csum_%7Bi%7D%7Bx_%7Bi%7D%3D1%7D)  
> **证：**  
> 必要性：假设 ![](https://www.zhihu.com/equation?tex=%5Cpi%3D%28%5Cpi_1%2C%5Cpi_2%2C...%29) 是平稳分布，则显然满足后 2 式，根据平稳分布的性质， ![](https://www.zhihu.com/equation?tex=%5Cpi_%7Bi%7D%3D%5Csum_%7Bj%7D%7Bp_%7Bij%7D%5Cpi_j%7D%2Ci%3D1%2C2%2C...) ，满足 1 式，得证。  
> 充分性：假设 ![](https://www.zhihu.com/equation?tex=%5Cpi%3D%28%5Cpi_1%2C%5Cpi_2%2C...%29) 为![](https://www.zhihu.com/equation?tex=X_%7Bt%7D)的分布，则：  
> ![](https://www.zhihu.com/equation?tex=P%28X_%7Bt%7D%3Di%29%3D%5Cpi_i%3D%5Csum_%7Bj%7D%7Bp_%7Bij%7D%5Cpi_j%7D%3D%5Csum_%7Bj%7D%7Bp_%7Bij%7DP%28X_%7Bt-1%7D%3Dj%29%7D%2Ci%3D1%2C2)  
> 所以![](https://www.zhihu.com/equation?tex=%5Cpi%3D%28%5Cpi_1%2C%5Cpi_2%2C...%29) 也为![](https://www.zhihu.com/equation?tex=X_%7Bt-1%7D)的分布，又因为对任意的 ![](https://www.zhihu.com/equation?tex=t) 成立，所以 ![](https://www.zhihu.com/equation?tex=%5Cpi%3D%28%5Cpi_1%2C%5Cpi_2%2C...%29) 是平稳分布。

## 2.2 马尔科夫链的性质

*   **不可约**

**一个不可约的马尔可夫链，从任意状态出发，当经过充分长时间后，可以到达任意状态。**数学语言是：

> 定义：给定一个马尔科夫链 ![](https://www.zhihu.com/equation?tex=X%3D%5Cleft%5C%7B+X_0%2CX_1%2C...%2CX_t%2C...+%5Cright%5C%7D) ，对于任意的状态 ![](https://www.zhihu.com/equation?tex=i%2Cj%5Cin+S) ，如果存在一个时刻 ![](https://www.zhihu.com/equation?tex=t) 满足： ![](https://www.zhihu.com/equation?tex=P%28X_t%3Di%7CX_%7B0%7D%3Dj%29%3E0) ，也就是说，时刻 ![](https://www.zhihu.com/equation?tex=0) 从状态![](https://www.zhihu.com/equation?tex=j) 出发，时刻 ![](https://www.zhihu.com/equation?tex=t) 到达状态 ![](https://www.zhihu.com/equation?tex=i) 的概率大于０，则称此马尔可夫链 ![](https://www.zhihu.com/equation?tex=X) 是不可约的 (irreducible)，否则称马尔可夫链是可约的 (reducible)。

![](https://pic2.zhimg.com/v2-d4c4260d8f88e053a58cec614cb56231_r.jpg)

<center>Fig. 8 可约的马尔科夫链</center>

*   **非周期**

> 定义：给定一个马尔科夫链 ![](https://www.zhihu.com/equation?tex=X%3D%5Cleft%5C%7B+X_0%2CX_1%2C...%2CX_t%2C...+%5Cright%5C%7D) ，对于任意的状态 ![](https://www.zhihu.com/equation?tex=i%5Cin+S) ，如果时刻 ![](https://www.zhihu.com/equation?tex=0) 从状态 ![](https://www.zhihu.com/equation?tex=i) 出发， ![](https://www.zhihu.com/equation?tex=t) 时刻返回状态的所有时间长 ![](https://www.zhihu.com/equation?tex=%5Cleft%5C%7B+t%3AP%28X_%7Bt%7D%3Di%7CX_%7B0%7D%3Di%29%3E0+%5Cright%5C%7D) 的最大公约数是 1，则称此马尔可夫链 ![](https://www.zhihu.com/equation?tex=X) 是非周期的，否则称马尔可夫链是周期的。

![](https://pic3.zhimg.com/v2-9a2a9e15e1ad954d76c834c598ef5376_r.jpg)

<center>FIg. 9 周期的马尔科夫链</center>

*   **定理：不可约且非周期的有限状态马尔可夫链，有唯一平稳分布存在。**

**定义：首达时间：**

![](https://www.zhihu.com/equation?tex=T_%7Bij%7D%3Dmin%5Cleft%5C%7B+n%3An%5Cgeq1%2CX_0%3Di%2CX_n%3Dj+%5Cright%5C%7D) 表示从状态 ![](https://www.zhihu.com/equation?tex=i) 出发首次到达状态 ![](https://www.zhihu.com/equation?tex=j) 的时间。若状态 ![](https://www.zhihu.com/equation?tex=i) 出发永远不能到达状态 ![](https://www.zhihu.com/equation?tex=j) ，则 ![](https://www.zhihu.com/equation?tex=T_%7Bij%7D%3D%2B%5Cinfty) 。

**定义：首达概率：**

![](https://www.zhihu.com/equation?tex=f_%7Bij%7D%5E%7B%28n%29%7D%3DP%28X_n%3Dj%2CX_m%5Cne+j%2Cm%3D1%2C2%2C...%2Cn-1%7CX_0%3Di%29)

![](https://www.zhihu.com/equation?tex=f_%7Bij%7D%5E%7B%28n%29%7D) 为从状态 ![](https://www.zhihu.com/equation?tex=i) 出发经过 ![](https://www.zhihu.com/equation?tex=n) 步首次到达状态 ![](https://www.zhihu.com/equation?tex=j) 的概率。

![](https://www.zhihu.com/equation?tex=f_%7Bij%7D%5E%7B%28%2B+%5Cinfty%29%7D) 为从状态 ![](https://www.zhihu.com/equation?tex=i) 出发永远不能到达状态 ![](https://www.zhihu.com/equation?tex=j) 的概率。

**定义：从状态 ![](https://www.zhihu.com/equation?tex=i) 出发经过有限步首次到达状态 ![](https://www.zhihu.com/equation?tex=j) 的概率：**

![](https://www.zhihu.com/equation?tex=f_%7Bij%7D%3D%5Csum_%7Bn%3D1%7D%5E%7B%2B%5Cinfty%7D%7Bf_%7Bij%7D%5E%7B%28n%29%7D%7D%3DP%28T_%7Bij%7D%3C%2B%5Cinfty%29)

**定义：**

**状态 ![](https://www.zhihu.com/equation?tex=i) 为常返态：** ![](https://www.zhihu.com/equation?tex=f_%7Bii%7D%3D1) ，即有限步一定能回来。

**状态** ![](https://www.zhihu.com/equation?tex=i) **为非常返态：**![](https://www.zhihu.com/equation?tex=f_%7Bii%7D%3C1) ，即有限步可能回不来。

**定义：平均返回时间：**

![](https://www.zhihu.com/equation?tex=%5Cmu_%7Bi%7D%3D%5Csum_%7Bn%3D1%7D%5E%7B%2B%5Cinfty%7D%7Bnf_%7Bii%7D%5E%7B%28n%29%7D%7D)

*   **正常返和零常返**

> **首先 ![](https://www.zhihu.com/equation?tex=i) 得是常返态，且若 ![](https://www.zhihu.com/equation?tex=%5Cmu_%7Bi%7D%3C%2B%5Cinfty) ，称状态 ![](https://www.zhihu.com/equation?tex=i) 为正常返。若** ![](https://www.zhihu.com/equation?tex=%5Cmu_%7Bi%7D%3D%2B%5Cinfty) **，称状态** ![](https://www.zhihu.com/equation?tex=i) **为零常返。**

*   **遍历态**

> ![](https://www.zhihu.com/equation?tex=i) 既是正常返又是非周期，就是遍历态。

![](https://pic1.zhimg.com/v2-4ac8de5653c49ccc2aede6dd9d535a18_r.jpg)

<center>Fig. 10 各种定义汇总</center>

直观上，一个正常返的马尔可夫链，其中任意一个状态，从其他任意一个状态出发，当时间趋于无穷时，首次转移到这个状态的概率不为 0。(从任意一个状态出发，走了能回来)

![](https://pic3.zhimg.com/v2-3941c61023e64b85b2ef2122924a878a_r.jpg)清华大学随机过程期末考试题：这个马尔科夫链是正常返的吗？

如上图所示的马尔科夫链，当 ![](https://www.zhihu.com/equation?tex=p%3Eq) 时是正常返的，当 ![](https://www.zhihu.com/equation?tex=p%3Cq) 时不是正常返的。

> 证：转移概率矩阵：  
> ![](https://www.zhihu.com/equation?tex=P%3D%5Cbegin%7Bbmatrix%7Dp+%26+p+%260%260%26...+%5C%5C+q+%26+0+%26p%260%26...+%5C%5C++0+%26+q+%260%26p%26...%5C%5C++0+%26+0+%26q%260%26...%5C%5C++...+%26+...+%26...%26...%26...%5Cend%7Bbmatrix%7D)  
> 若达到了平稳分布，则 ![](https://www.zhihu.com/equation?tex=P%5Cpi%3D%5Cpi) ，代入 ![](https://www.zhihu.com/equation?tex=P) 化简，且注意 ![](https://www.zhihu.com/equation?tex=p%2Bq%3D1) 。  
> ![](https://www.zhihu.com/equation?tex=%5Cpi_2%3D%5Cfrac%7Bq%7D%7Bp%7D%5Cpi_1%2C%5Cpi_3%3D%5Cfrac%7Bq%7D%7Bp%7D%5Cpi_2%2C%5Cpi_4%3D%5Cfrac%7Bq%7D%7Bp%7D%5Cpi_3%2C...%5C%5C+%5Cpi_1%2B%5Cpi_2%2B%5Cpi_3%2B...%3D1)  
> 当 ![](https://www.zhihu.com/equation?tex=p%3Eq) 时，平稳分布是： ![](https://www.zhihu.com/equation?tex=%5Cpi_i%3D%28%5Cfrac%7Bq%7D%7Bp%7D%29%5E%7Bi%7D%28%5Cfrac%7Bp-q%7D%7Bp%7D%29%2Ci%3D1%2C2%2C...)  
> 当时间趋于无穷时，转移到任何一个状态的概率不为 0，马尔可夫链是正常返的。  
> 当 ![](https://www.zhihu.com/equation?tex=p%3Cq) 时，不存在平稳分布，马尔可夫链不是正常返的。  
> 你看，清华的期末考试也不过如此~

*   **遍历定理：不可约、非周期且正常返的马尔可夫链，有唯一平稳分布存在，并且转移概率的极限分布是马尔可夫链的平稳分布。**

![](https://www.zhihu.com/equation?tex=%5Clim_%7Bt+%5Crightarrow+%2B%5Cinfty+%7D%7BP%28X_t%3Di%7CX_0%3Dj%29%3D%5Cpi_i%7D%2Ci%3D1%2C2%2C...%3Bj%3D1%2C2%2C...)

你会发现，不可约、非周期且正常返的马尔可夫链，它的转移概率矩阵在 ![](https://www.zhihu.com/equation?tex=t+%5Crightarrow+%2B%5Cinfty) 时竟然是：

![](https://www.zhihu.com/equation?tex=%5Cunderset%7Bn%5Crightarrow+%5Cinfty%7D%7Blim%7DP%5E%7Bn%7D%3D%5Cbegin%7Bbmatrix%7D%5Cpi%281%29+%26%5Cpi%281%29+%26%5Ccdots+%26%5Cpi%281%29+%26+%5Ccdots+%5C%5C+%5Cpi%282%29+%26%5Cpi%282%29+%26%5Ccdots+%26%5Cpi%282%29+%26+%5Ccdots+%5C%5C+%5Cvdots+%26+%5Cvdots+%26+%5Cvdots+%26%5Cddots+%26%5Cvdots+%5C%5C+%5Cpi%28j%29+%26%5Cpi%28j%29+%26%5Ccdots+%26%5Cpi%28j%29+%26+%5Ccdots+%5C%5C+%5Cvdots+%26+%5Cvdots+%26+%5Cvdots+%26+%5Cddots+%26+%5Cvdots+%5Cend%7Bbmatrix%7D)

> **今后，你如果再遇到某个题目说 “一个满足遍历定理的马尔科夫链，...” 你就应该立刻意识到这个马尔科夫链只有一个平稳分布，而且它就是转移概率的极限分布。且随机游走的起始点并不影响得到的结果，即从不同的起始点出发，都会收敛到同一平稳分布。**注意这里很重要，一会要考。  

说了这么多概念，用大白话做个总结吧：

*   不可约：每个状态都能去到。**(遍历别人)**
*   非周期：返回时间公约数是 1。**(不能周期性遍历，保证遍历的公平性)**
*   正常返：离开此状态有限步一定能回来。迟早会回来。**(遍历自己)**
*   零常返：离开此状态能回来，但需要无穷多步。
*   非常返：离开此状态有限步不一定回得来。
*   遍历定理：不可约，非周期，正常返 ![](https://www.zhihu.com/equation?tex=%5Crightarrow) 有唯一的平稳分布。

*   **可逆马尔可夫链**

> 定义：给定一个马尔科夫链 ![](https://www.zhihu.com/equation?tex=X%3D%5Cleft%5C%7B+X_0%2CX_1%2C...%2CX_t%2C...+%5Cright%5C%7D) ，如果有状态分布 ![](https://www.zhihu.com/equation?tex=%28%5Cpi_1%2C%5Cpi_2%2C%5Cpi_3%2C...%29) 。对于任意的状态 ![](https://www.zhihu.com/equation?tex=i%2Cj%5Cin+S)，对任意一个时刻 ![](https://www.zhihu.com/equation?tex=t) 满足：  
> ![](https://www.zhihu.com/equation?tex=P%28X_t%3Di%7CX_%7Bt-1%7D%3Dj%29%5Cpi_j%3DP%28X_t%3Dj%7CX_%7Bt-1%7D%3Di%29%5Cpi_i%2Ci%3D1%2C2%2C...)  
> 或简写为：  
> ![](https://www.zhihu.com/equation?tex=p_%7Bij%7D%5Cpi_j%3Dp_%7Bji%7D%5Cpi_i)  
> 则称此马尔可夫链 ![](https://www.zhihu.com/equation?tex=X) 为可逆马尔可夫链 (reversible Markov chain)，上式称为  
> **细致平衡方程 (detailed balance equation)**。  
> 直观上，如果有可逆的马尔可夫链，那么以该马尔可夫链的**平稳分布**作为**初始分布**，进行随机状态转移，无论是面向未来还是面向过去，**任何一个时刻**的状态分布都是该**平稳分布**。概率分布![](https://www.zhihu.com/equation?tex=%5Cpi)是状态转移矩阵![](https://www.zhihu.com/equation?tex=P)的平稳分布。  
> **定理：**满足细致平衡方程的状态分布 ![](https://www.zhihu.com/equation?tex=%5Cpi) 就是该马尔可夫链的平稳分布 ![](https://www.zhihu.com/equation?tex=P%5Cpi%3D%5Cpi) 。  
> **证：**  
> ![](https://www.zhihu.com/equation?tex=%28P%5Cpi%29_%7Bi%7D%3D%5Csum_%7Bj%7D%7Bp_%7Bij%7D%5Cpi_%7Bj%7D%7D%3D%5Csum_%7Bj%7D%7Bp_%7Bji%7D%5Cpi_%7Bi%7D%7D%3D%5Cpi_%7Bi%7D+%5Csum_%7Bj%7D%7Bp_%7Bji%7D%7D%3D%5Cpi_%7Bi%7D%2Ci%3D1%2C2%2C...)

以上就是关于蒙特卡罗方法和马尔科夫链你分别需要掌握的知识，说了这么久还没有进入正题。本文是为了让你打好基础，那从下一篇文章开始，我们会讲解什么是马尔科夫链蒙特卡罗方法以及它的具体细节。



# :apple:马尔科夫蒙特卡洛的结合

## 3.1 介绍

以上内容的核心思想可以用下图概括：

![](https://pic2.zhimg.com/v2-deb9b905eeaae63014123663fb3089b9_r.jpg)

一般的采样问题，以及期望求解，数值近似问题，蒙特卡罗方法都能很好地解决；但遇到多元变量的随机分布以及复杂的概率密度时，仅仅使用蒙特卡罗方法就会显得捉襟见肘，这时就需要这篇文章要讲的马尔可夫链蒙特卡罗法来解决这个问题了。我们先从一维的讲起：

在开始之前首先统一下定义：

我们用符号 ![](https://www.zhihu.com/equation?tex=%5Cpi_%7Bi%7D%3D%5Cpi%28i%29%3D%5Clim_%7Bt+%5Crightarrow+%2B%5Cinfty%7D%7BP%28X_t%3Di%29%7D) 代表一个概率，即马尔科夫链达到平稳分布的时候，状态位于第 ![](https://www.zhihu.com/equation?tex=i) 个状态的概率。

**马尔科夫链和蒙特卡罗方法是如何结合在一起的？**

**一张图解释清楚：**

![](https://pic3.zhimg.com/v2-702c2ea7a4f9c53282cf10a5c9098aee_r.jpg)

还记得上篇文章中提到的遍历定理吗？如果你忘记了这些定义，请打开上面的链接再复习一遍并点赞：

*   **遍历定理：不可约、非周期且正常返的马尔可夫链，有唯一平稳分布存在，并且转移概率的极限分布是马尔可夫链的平稳分布。**

![](https://www.zhihu.com/equation?tex=%5Clim_%7Bt+%5Crightarrow+%2B%5Cinfty+%7D%7BP%28X_t%3Di%7CX_0%3Dj%29%3D%5Cpi_i%7D%2Ci%3D1%2C2%2C...%3Bj%3D1%2C2%2C...)

当时让你记了一句话：**今后，你如果再遇到某个题目说 “一个满足遍历定理的马尔科链，...” 你就应该立刻意识到这个马尔科夫链只有一个平稳分布，而且它就是转移概率的极限分布。且随机游走的起始点并不影响得到的结果，即从不同的起始点出发，都会收敛到同一平稳分布。有一个成语叫殊途同归，形容的就是这件事。**

所以，我先定义一个满足遍历定理的马尔可夫链![](https://www.zhihu.com/equation?tex=X%3D%5Cleft%5C%7B+X_0%2CX_1%2C...%2CX_t%2C...+%5Cright%5C%7D)，每一个状态就代表我在**王者荣耀**里面玩哪个英雄，比如初始状态 ![](https://www.zhihu.com/equation?tex=X_0%3D1) 就是玩射手，初始状态 ![](https://www.zhihu.com/equation?tex=X_0%3D3) 就是玩打野等等。

现在这个马尔科夫链因为满足遍历定理，所以有个平稳分布，代表的就是当 ![](https://www.zhihu.com/equation?tex=t%5Crightarrow+%2B%5Cinfty) 时，我选择哪一个英雄的概率分布。

![](https://pic4.zhimg.com/v2-76e3417a542d0b54676ae1b878d1c197_r.jpg)

**现在重复一下要解决的问题：**

1. 从一个目标分布 ![](https://www.zhihu.com/equation?tex=p%28x%29) 中进行抽样；
2. 求出 ![](https://www.zhihu.com/equation?tex=f%28x%29) 的数学期望![](https://www.zhihu.com/equation?tex=E_%7Bx%5Csim+p%28x%29%7D%5Bf%28x%29%5D)，那我就可以假设：

**王者荣耀马尔可夫链的平稳分布 = 目标分布 ![](https://www.zhihu.com/equation?tex=p%28x%29)** 

**所以，一个惊人的结论诞生了：**

在王者荣耀这个马尔科夫链上游走 1 次，对应的状态就相当于是从目标分布 ![](https://www.zhihu.com/equation?tex=p%28x%29) 中进行抽样 1 个样本。换句话说，比如王者荣耀平稳分布是 ![](https://www.zhihu.com/equation?tex=%5B0.5%2C0.2%2C0.3%5D) ，目标分布 ![](https://www.zhihu.com/equation?tex=p%28x%29)也应该是 ![](https://www.zhihu.com/equation?tex=%5B0.5%2C0.2%2C0.3%5D)，那在马尔科夫链上游走 1 次，比如说这把选了辅助，就相当于是在![](https://www.zhihu.com/equation?tex=p%28x%29)中采样了 ![](https://www.zhihu.com/equation?tex=x_2) 。

**所以，每个时刻在这个马尔可夫链上进行随机游走一次，就可以得到一个样本。根据遍历定理，当时间趋于无穷时，样本的分布趋近平稳分布，样本的函数均值趋近函数的数学期望。**

所以，当时间足够长时 (时刻大于某个正整数 ![](https://www.zhihu.com/equation?tex=m) )，在之后的时间 (时刻小于等于某个正整数 ![](https://www.zhihu.com/equation?tex=n%2Cn%3Em) ) 里随机游走得到的样本集合 ![](https://www.zhihu.com/equation?tex=%5Cleft%5C%7B+x_%7Bm%2B1%7D%2C...%2Cx_%7Bn%7D+%5Cright%5C%7D) 就是目标概率分布的抽样结果，得到的函数均值 (遍历均值) 就是要计算的数学期望值：

![](https://www.zhihu.com/equation?tex=%5Chat+Ef%3D%5Cfrac%7B1%7D%7Bn-m%7D%5Csum_%7Bi%3Dm%2B1%7D%5E%7Bn%7D%7Bf%28x_i%29%7D)

到时刻 ![](https://www.zhihu.com/equation?tex=m) 为止的时间段称为燃烧期。

马尔可夫链蒙特卡罗法中得到的样本序列，相邻的样本点是相关的，而不是独立的。因此，在**需要独立样本**时，可以在该样本序列中再次进行随机抽样，比如每隔一段时间取一次样本，将这样得到的子样本集合作为独立样本集合。马尔可夫链蒙特卡罗法比接受 - 拒绝法更容易实现，因为只需要定义马尔可夫链，而不需要定义建议分布。一般来说马尔可夫链蒙特卡罗法比接受 - 拒绝法效率更高，没有大量被拒绝的样本，虽然燃烧期的样本也要抛弃。

**最大的问题是：给了我目标分布 ![](https://www.zhihu.com/equation?tex=p%28x%29)，对应的王者荣耀马尔可夫链怎么构造？假设收敛的步数为 ![](https://www.zhihu.com/equation?tex=m) ，即迭代了 ![](https://www.zhihu.com/equation?tex=m) 步之后收敛，那![](https://www.zhihu.com/equation?tex=m)是多少？迭代步数 ![](https://www.zhihu.com/equation?tex=n)又是多少？**

常用的马尔可夫链蒙特卡罗法有 Metropolis-Hastings 算法、吉布斯抽样。

**马尔可夫链蒙特卡罗方法的基本步骤是：**

1.  构造一个王者荣耀马尔可夫链，使其平稳分布为目标分布 ![](https://www.zhihu.com/equation?tex=p%28x%29) 。
2.  从某个初始状态 ![](https://www.zhihu.com/equation?tex=x_%7B0%7D) 出发，比如第一把选射手，用构造的马尔可夫链随机游走，产生样本序列 ![](https://www.zhihu.com/equation?tex=x_%7B0%7D%2Cx_%7B1%7D%2C...%2Cx_%7Bt%7D%2C...)
3.  应用遍历定理，确定正整数 ![](https://www.zhihu.com/equation?tex=m) 和 ![](https://www.zhihu.com/equation?tex=n) ，得到平稳分布的样本集合： ![](https://www.zhihu.com/equation?tex=%5Cleft%5C%7B+x_%7Bm%2B1%7D%2Cx_%7Bm%2B2%7D%2C...%2Cx_%7Bn%7D%5Cright%5C%7D) ，求得函数 ![](https://www.zhihu.com/equation?tex=f%28x%29) 的均值：

![](https://www.zhihu.com/equation?tex=%5Chat+Ef%3D%5Cfrac%7B1%7D%7Bn-m%7D%5Csum_%7Bi%3Dm%2B1%7D%5E%7Bn%7D%7Bf%28x_i%29%7D)

## 3.2 如何构造一个王者荣耀马尔可夫链？

比如说现在已知的目标分布 ![](https://www.zhihu.com/equation?tex=p%28x%29)是 ![](https://www.zhihu.com/equation?tex=%5B0.5%2C0.2%2C0.3%5D)，想构造一个马尔可夫链，使它的平稳分布也是 ![](https://www.zhihu.com/equation?tex=%5B0.5%2C0.2%2C0.3%5D)，关键还是要求出状态转移矩阵：

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Bbmatrix%7D0.5%5C%5C0.2%5C%5C0.3%5Cend%7Bbmatrix%7D%3DP%5Cbegin%7Bbmatrix%7D0.5%5C%5C0.2%5C%5C0.3%5Cend%7Bbmatrix%7D)

在上篇文章中提到过这样一个定理：

**定理：**满足细致平衡方程的状态分布![](https://www.zhihu.com/equation?tex=%5Cpi)就是该马尔可夫链的平稳分布![](https://www.zhihu.com/equation?tex=P%5Cpi%3D%5Cpi)。

所以，我只需要找到**可以使状态分布![](https://www.zhihu.com/equation?tex=%5Cpi)满足细致平衡方程的矩阵 ![](https://www.zhihu.com/equation?tex=P) 即可**，即满足：

![](https://www.zhihu.com/equation?tex=%CF%80_iP_%7Bj%2Ci%7D%3D%CF%80_jP_%7Bi%2Cj%7D)

**上面这个细致平衡条件(Detailed Balance Condition)说的是任意两个状态之间转移是等可能的。**需要注意的是这是一个充分条件，而**不是必要条件**，也就是说存在具有平稳分布的马尔科夫链不满足此细致平衡条件。

![image-20201120114801183](C:\Users\LapTop-of-ChenWei\AppData\Roaming\Typora\typora-user-images\image-20201120114801183.png)

注意上式的证明摘自[原文](https://blog.csdn.net/guolindonggld/article/details/79597491)，其中$P_{ij}$的定义等价于上文说到的$P_{ji}$。

仅仅从细致平衡方程还是很难找到合适的矩阵 ![](https://www.zhihu.com/equation?tex=P) 。比如我们的目标平稳分布是 ![](https://www.zhihu.com/equation?tex=%CF%80) , 随机找一个马尔科夫链状态转移矩阵 ![](https://www.zhihu.com/equation?tex=Q) , 它是很难满足细致平衡方程，即：

![](https://www.zhihu.com/equation?tex=%CF%80_iQ_%7Bj%2Ci%7D%5Cne%CF%80_jQ_%7Bi%2Cj%7D)

那么如何使这个等式满足呢？下面我们来看 MCMC 采样如何解决这个问题。

## 3.3 马尔科夫链蒙特卡罗方法总论

### 3.3.1 MCMC算法流程概述

构造一个 ![](https://www.zhihu.com/equation?tex=%5Calpha_%7Bij%7D) 和 ![](https://www.zhihu.com/equation?tex=%5Calpha_%7Bji%7D) ，使上式强制取等号，即：

![](https://www.zhihu.com/equation?tex=%CF%80_iQ_%7Bj%2Ci%7D%5Calpha_%7Bj%2Ci%7D%3D%CF%80_jQ_%7Bi%2Cj%7D%5Calpha_%7Bi%2Cj%7D)

要使上式恒成立，只需要取：

![](https://www.zhihu.com/equation?tex=%5Calpha_%7Bj%2Ci%7D%3D%CF%80_jQ_%7Bi%2Cj%7D%5C%5C%5Calpha_%7Bi%2Cj%7D%3D%CF%80_iQ_%7Bj%2Ci%7D)

所以，马尔可夫链的状态转移矩阵就呼之欲出了：

![](https://www.zhihu.com/equation?tex=P_%7Bj%2Ci%7D%3DQ_%7Bj%2Ci%7D%5Calpha_%7Bj%2Ci%7D%5C%5CP_%7Bi%2Cj%7D%3DQ_%7Bi%2Cj%7D%5Calpha_%7Bi%2Cj%7D)

咦，状态转移矩阵 ![](https://www.zhihu.com/equation?tex=Q) 是我们胡乱设的， ![](https://www.zhihu.com/equation?tex=%5Calpha) 值是可以根据 ![](https://www.zhihu.com/equation?tex=Q) 和目标分布 ![](https://www.zhihu.com/equation?tex=p%28x%29)算出来的，然后，要构造的满足细致平衡方程的矩阵![](https://www.zhihu.com/equation?tex=P)竟然被我们求出来了！！！

![](https://www.zhihu.com/equation?tex=%5Calpha_%7Bi%2Cj%7D) 的专业术语叫做接受率。取值在 ![](https://www.zhihu.com/equation?tex=%5B0%2C1%5D) 之间，可以理解为一个概率值。

状态转移矩阵 ![](https://www.zhihu.com/equation?tex=Q)的平稳分布专业术语叫做建议分布 (proposal distribution)。

**我们回顾一下在一顿数学公式之后，问题的演变过程：**

从目标分布 ![](https://www.zhihu.com/equation?tex=p%28x%29)中采样

![](https://www.zhihu.com/equation?tex=%5Crightarrow) 在马尔科夫链 (状态转移矩阵为 ![](https://www.zhihu.com/equation?tex=P) ) 中随机游走，当达到平稳分布时，每个时刻随机游走一次，就可以得到一个样本。

![](https://www.zhihu.com/equation?tex=%5Crightarrow) 在马尔科夫链 (状态转移矩阵为 ![](https://www.zhihu.com/equation?tex=Q) ) 中随机游走，当达到平稳分布时，每个时刻随机游走一次，就可以得到一个样本，这时，以一定的接受率获得，和上篇文章中的拒绝 - 接受采样极其类似，以一个常见的马尔科夫链状态转移矩阵 ![](https://www.zhihu.com/equation?tex=Q) 通过一定的接受 - 拒绝概率得到目标转移矩阵 ![](https://www.zhihu.com/equation?tex=P) 。

好了，现在我们来总结下 MCMC 的采样过程。

　　　　1）输入我们任意选定的马尔科夫链状态转移矩阵 $Q$，平稳分布$π(x)$，设定状态转移次数阈值 $n_1$，需要的样本个数 $n_2$

　　　　2）从任意简单概率分布采样得到初始状态值 $x_0$

　　　　3）for $t=0$ to $n_1+n_2−1$: 

　　　　　　a) 从条件概率分布 $Q(x|x_t)$中采样得到样本 $x_∗$

　　　　　　b) 从均匀分布采样 $u∼uniform[0,1]$

　　　　　　c) 如果 $u<α(x_t,x_∗)=π(x_∗)Q(x_∗,x_t)$ , 则接受转移 $x_t→x_∗$，即 $x_{t+1}=x_∗$

　　　　　　d) 否则不接受转移，即 $x_{t+1}=x_t$

　　　　样本集 $(x_{n_1},x_{n_{1+1}},...,x_{n_1+n_2−1})$即为我们需要的平稳分布对应的样本集。

![mcmc-algo-1](http://cos.name/wp-content/uploads/2013/01/mcmc-algo-1.jpg)

### 3.3.2 Metropolis-Hastings 采样算法

> 主要改进了接收率$\alpha$，使其不至于总是很小导致总是很难接受样本

M-H 采样是 Metropolis-Hastings 采样的简称，这个算法首先由 Metropolis 提出，被 Hastings 改进，因此被称之为 Metropolis-Hastings 采样或 M-H 采样

M-H 采样解决了我们上一节 MCMC 采样接受率过低的问题。

我们回到 MCMC 采样的细致平稳条件：

$$
π(i)Q(i,j)α(i,j)=π(j)Q(j,i)α(j,i)
$$
我们采样效率低的原因是$α(i,j)$太小了，比如为 0.1，而$α(j,i)$为 0.2。即：

$$
π(i)Q(i,j)×0.1=π(j)Q(j,i)×0.2
$$
这时我们可以看到，如果两边同时扩大五倍，接受率提高到了 0.5，但是细致平稳条件却仍然是满足的，因此核心公式还是没有变，即：

$$
π(i)Q(i,j)×0.5=π(j)Q(j,i)×1
$$
这样我们的接受率可以做如下改进，即：

$$
α(i,j)=min\left\{\frac{π(j)Q(j,i)}{π(i)Q(i,j)},1\right\}
$$
通过这个微小的改造，我们就得到了可以在实际应用中使用的 M-H 采样算法过程如下：

- 1）输入我们任意选定的马尔科夫链状态转移矩阵 $Q$，平稳分布$π(x)$，设定状态转移次数阈值 $n1$，需要的样本个数 $n2$

- 2）从任意简单概率分布采样得到初始状态值 $x0$

- 3）for $t=0$ to $n_1+n_2−1$: 

  - a) 从条件概率分布 $Q(x|x_t)$中采样得到样本 $x_∗$

  - b) 从均匀分布采样 $u∼uniform[0,1]$

  - c) 如果 $u<α(x_t,x_∗)=min\left\{\frac{π(j)Q(j,i)}{π(i)Q(i,j)},1\right\}$, 则接受转移 $x_t→x_∗$，即 $x_{t+1}=x_∗$
  - d) 否则不接受转移，即 $x_{t+1}=x_t$

样本集 $(x_{n_1},x_{n_1+1},...,x_{n_1+n_2−1})$即为我们需要的平稳分布对应的样本集。

很多时候，我们选择的马尔科夫链状态转移矩阵 $Q$如果是对称的，即满足 $Q(i,j)=Q(j,i)$, 这时我们的接受率可以进一步简化为：

$$
α(i,j)=min\left\{\frac{π(j)}{π(i)},1\right\}
$$
![mcmc-algo-2](http://cos.name/wp-content/uploads/2013/01/mcmc-algo-2.jpg)

**M-H 采样的 Python 实现**

- 例子1：假设目标平稳分布是一个均值 10，标准差 5 的正态分布，而选择的马尔可夫链状态转移矩阵 ![](https://www.zhihu.com/equation?tex=Q_%7Bj%2Ci%7D) 的条件转移概率是以 ![](https://www.zhihu.com/equation?tex=i) 为均值, 方差 1 的正态分布在位置 ![](https://www.zhihu.com/equation?tex=j) 的值。

```python
import random
from scipy.stats import norm    # scipy.stats专门用来生成指定分布
import matplotlib.pyplot as plt   # 绘图用的

"""
    stats连续型随机变量的公共方法
    rvs：产生服从指定分布的随机数, random variates of given type
    pdf：概率密度函数
    cdf：累计分布函数
    sf：残存函数（1-CDF）
    ppf：分位点函数（CDF的逆）
    isf：逆残存函数（sf的逆）
    fit：对一组随机取样进行拟合，最大似然估计方法找出最适合取样数据的概率密度函数系数。
    *离散分布的简单方法大多数与连续分布很类似，但是pdf被更换为密度函数pmf。
    
    常见分布 stats.
    beta：beta分布
    f：F分布
    gamma：gam分布
    poisson：泊松分布
    hypergeom：超几何分布
    lognorm：对数正态分布
    binom：二项分布
    uniform：均匀分布
    chi2：卡方分布
    cauchy：柯西分布
    laplace：拉普拉斯分布
    rayleigh：瑞利分布
    t：学生T分布
    norm：正态分布
    expon：指数分布
"""

def norm_dist_prob(x):
    y = norm.pdf(x, loc=10, scale=5)  # mu=10,std=5的正态分布在x处的概率密度
    return y

T = 5000  # 采样5000次
pi = [0 for i in range(T)]   # 存储获取的样本，初始状态为0
sigma = 1   # 随机游走的幅度
t = 0
while t < T-1:
    t += 1
    
    ## Step 1: 随机游走
    # 以上一状态为中心随机游走得到下一个状态,size设定生成的数组形状与大小
    pi_new = norm.rvs(loc=pi[t-1], scale= sigma, size=1)  
    # 计算接受率，状态转移矩阵为对称的
    alpha = min(1, (norm_dist_prob(pi_new[0]) / norm_dist_prob(pi[t-1])))  
    
    ## Step 2: 判断游走的合理性
    u = random.uniform(0,1)
    pi[t] = pi_star[0] if u < alpha else pi[t-1]
    
plt.scatter(pi, norm.pdf(pi, loc=10, scale=5), label='Target Distribution', c='red')
num_bins = 50   # 直方图有50个块块
plt.hist(pi, num_bins, density=1, facecolor='green', alpha=0.7, label='Samples Distribution')
plt.legend()
plt.show()
```

![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXoAAAD4CAYAAADiry33AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3dfXTU1b3v8fc3CQgWKhSwBwEJelB5iAQbShDFeBALSOReqlcRVGorRUQ92mOl9SylPXXVVdEqS8Si9aCXGG31lAJlWUUBtdVKwChPPqBFCHjLg6JQUJ72/WMmMZnM5Lcnmck8fV5rZZGZ356Z7wwz3+zZv72/25xziIhI9spLdQAiIpJcSvQiIllOiV5EJMsp0YuIZDklehGRLFeQ6gCi6dq1qyssLEx1GCIiGWPNmjW7nXPdoh1Ly0RfWFhIVVVVqsMQEckYZvZRrGMauhERyXJK9CIiWU6JXkQky6XlGH00hw8fpqamhi+++CLVoUiaa9euHT179qRNmzapDkUkLWRMoq+pqaFjx44UFhZiZqkOR9KUc449e/ZQU1NDnz59Uh2OSFrImKGbL774gi5duijJS5PMjC5duuibn0g9GZPoASV58aL3iUhDGZXoRUQkfhkzRh+pvLI8ofe3ZOKSwDZ33XUXTz75JPn5+eTl5fGb3/yGoUOHJjSO+srKypg9ezYlJSXNvo8FCxZw66230rNnT/bv388pp5zCnXfeydlnnw3AHXfcwYgRI7jgggui3n7RokWcdtpp9O/fP+rxhx9+mOOPP56rrroq7nj37t3Lk08+yfTp0wHYsWMHN954I88880wznqmIxJKxib61vfbaayxdupS1a9dy3HHHsXv3bg4dOpTqsLxcdtllPPjggwCsWLGCCRMmsGLFCvr168fPf/7zJm+7aNEixo0bFzXRHzlyhGnTpjU7rr179/LQQw/VJfqTTjopZ5J8rI6KT4dDJF4auvH08ccf07VrV4477jgAunbtykknnQTAz3/+c4YMGcLAgQOZOnUqtbt2lZWVcfPNNzNixAj69evH6tWrmTBhAn379uU///M/AdiyZQtnnHEGV199NWeeeSaXXHIJBw4caPT4zz//PMOGDeOss87i0ksvZf/+/QDMnDmT/v37c+aZZ/If//Efgc/j/PPPZ+rUqcyfPx+AKVOm1CXXyPv661//yuLFi7n11lspLi7mgw8+oKysjJ/+9Kecd955PPDAA8yaNYvZs2fX3f/ChQs5++yzGThwIG+88QZAozYDBw5ky5YtzJw5kw8++IDi4mJuvfVWtmzZwsCBA4HQyffvfe97FBUVMXjwYFasWAGEvqFMmDCB0aNH07dvX3784x/7/heK5Cwlek8XXngh27Zt47TTTmP69OmsWrWq7tiMGTNYvXo169ev5+DBgyxdurTuWNu2bXn55ZeZNm0a48ePZ+7cuaxfv54FCxawZ88eAN59912mTp3K22+/zde//nUeeuihBo+9e/dufvGLX7B8+XLWrl1LSUkJ9913H5988gl/+MMf2LBhA2+//XbdH48gZ511Fu+8806D66Ld19lnn83FF1/MPffcQ3V1NaeeeioQ6omvWrWKH/3oR43u+5///Cd//etfeeihh7jmmmuajOPuu+/m1FNPpbq6mnvuuafBsblz5wKwbt06Kisrufrqq+tm0lRXV/P000+zbt06nn76abZt2+b1vEVylYZuPHXo0IE1a9bwyiuvsGLFCi677DLuvvtupkyZwooVK/jVr37FgQMH+OSTTxgwYADl5aGv5hdffDEARUVFDBgwgO7duwNwyimnsG3bNjp16kSvXr0YPnw4AJMnT2bOnDkNeuevv/46GzdurGtz6NAhhg0bxte//nXatWvHD37wAy666CLGjRvn9Vyi7RMcz31ddtllMY9NnDgRgBEjRvD555+zd+9er5givfrqq9xwww0AnHHGGfTu3Zv33nsPgJEjR3LCCScA0L9/fz766CN69erVrMdJtkSfSxJpDiX6OOTn51NWVkZZWRlFRUU8/vjjXH755UyfPp2qqip69erFrFmzGszhrh3qycvLq/u99vKRI0eAxtMBIy875xg1ahSVlZWNYnrjjTd48cUXeeqpp3jwwQd56aWXAp/Hm2++Sb9+/RpcV1BQ4H1fX/va12Led7TnUlBQwLFjx+qu85nj3tSm9fVfx/z8/LrXUUSi09CNp3fffZf333+/7nJ1dTW9e/euS1pdu3Zl//79zTqZuHXrVl577TUAKisrOeeccxocLy0t5S9/+QubN28G4MCBA7z33nvs37+fzz77jLFjx3L//fdTXV0d+FirVq1i/vz5XHvttQ2uj3VfHTt2ZN++fd7P5emnnwZCPfITTjiB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)

- 例子2：再假设目标平稳分布是一个 ![](https://www.zhihu.com/equation?tex=a%3D2.37%2Cb%3D0.627) 的 ![](https://www.zhihu.com/equation?tex=%5Cbeta) 分布，而选择的马尔可夫链状态转移矩阵 ![](https://www.zhihu.com/equation?tex=Q_%7Bj%2Ci%7D) 的条件转移概率是以 ![](https://www.zhihu.com/equation?tex=i) 为均值, 方差 1 的正态分布在位置 ![](https://www.zhihu.com/equation?tex=j) 的值。

![](https://pic2.zhimg.com/v2-4222fc2567b1f9e7405d688ac6ebaff1_b.jpg)

```python
import random
from scipy.stats import beta, norm
import matplotlib.pyplot as plt

def beta_dist_prob(x):
    a = 2.37
    b = 0.627
    y = beta(a, b).pdf(x)
    return y
    
T = 5000
pi = [0 for i in range(T)]
sigma = 1
t = 0
while t < T-1:
    t += 1
    
    pi_new = norm.rvs(loc=pi[t-1], scale=sigma, size=1)
    alpha = min(1, beta_dist_prob(pi_new[0]) / beta_dist_prob(pi[t-1]))
    
    u = random.uniform(0, 1)
    pi[t] = pi_new[0] if u < alpha else pi[t-1]
    
plt.scatter(pi, beta_dist_prob(pi), label="Target Distribution", c='red')
num_bins = 50
plt.hist(pi, num_bins, density=1, facecolor='green', alpha=0.7, label='Samples Distribution')
plt.legend()
plt.show()
```

![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXoAAAD4CAYAAADiry33AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3de3RU5f3v8feXBAUURSEqAiVoUblEggZBUIzHSwFR+kM5gpdCvaSIVKuthdYusFZXPQe81B+gjdUGlxg41UrVH23xQkWqlgQaBEQRNEgaFwRYKBRU0O/5YybpEGaSycwkk+x8XmvNmtnP3vvZz2aHT3ae2fvZ5u6IiEhwtUl3A0REpHEp6EVEAk5BLyIScAp6EZGAU9CLiARcZrobEE2XLl08Ozs73c0QEWkxVq1atcPds6LNa5ZBn52dTWlpabqbISLSYpjZlljz1HUjIhJwCnoRkYBT0IuIBFyz7KOP5sCBA1RUVPDFF1+kuynSzLVr147u3bvTtm3bdDdFpFloMUFfUVFBx44dyc7OxszS3RxpptydnTt3UlFRQa9evdLdHJFmocV03XzxxRd07txZIS91MjM6d+6sv/xEIrSYM3pAIS9x0c+JtDjRfmZTOLJwizmjFxEJpFgnJik8YWlRZ/SRLi++PKX1vTThpXqXuf/++3n22WfJyMigTZs2/Pa3v2Xw4MEpbUek/Px8Zs+eTV5eXsJ1FBUVcdddd9G9e3f27t3LKaecwsyZMxk6dCgAM2bMYPjw4Vx88cVR11+8eDGnnXYaffv2jTr/8ccfp0OHDnzve99rcHt3797Ns88+y5QpUwCorKzktttu47nnnktgT0UklhYb9E3t7bff5uWXX2b16tUceeSR7Nixg6+++irdzYrL1VdfzZw5cwBYtmwZY8eOZdmyZfTp04d77723znUXL17M6NGjowb9wYMHmTx5csLt2r17N/PmzasJ+pNPPlkhL9II1HUTp08//ZQuXbpw5JFHAtClSxdOPvlkAO69914GDRpE//79KSgooPqpXfn5+dxxxx0MHz6cPn36UFJSwtixY+nduze/+MUvACgvL+eMM85g4sSJnHnmmVx11VXs27fvsO0vXbqUc889l7POOotx48axd+9eAKZPn07fvn0588wz+clPflLvflx44YUUFBRQWFgIwKRJk2rCtXZdb731Fi+++CJ33XUXubm5bN68mfz8fH7+859zwQUX8Jvf/IZ77rmH2bNn19T/zDPPMHToUPr378/KlSsBDlumf//+lJeXM336dDZv3kxubi533XUX5eXl9O/fHwh9+f7973+fnJwcBg4cyLJly4DQXyhjx45lxIgR9O7dm5/+9KfxHkKRVktBH6dLL72UrVu3ctpppzFlyhTeeOONmnlTp06lpKSEdevWsX//fl5++eWaeUcccQTLly9n8uTJjBkzhrlz57Ju3TqKiorYuXMnAB988AEFBQW8++67HHPMMcybN++Qbe/YsYP77ruPV199ldWrV5OXl8dDDz3Erl27eOGFF1i/fj3vvvtuzS+P+px11lm8//77h5RFq2vo0KFcccUVzJo1i7KyMk499VQgdCb+xhtv8OMf//iwuv/973/z1ltvMW/ePG644YY62/HAAw9w6qmnUlZWxqxZsw6ZN3fuXADWrl1LcXExEydOrLmSpqysjEWLFrF27VoWLVrE1q1b49pvkdZKQR+no48+mlWrVlFYWEhWVhZXX301RUVFQKg7ZPDgweTk5PD666+zfv36mvWuuOIKAHJycujXrx9du3blyCOP5JRTTqkJqB49ejBs2DAArrvuOlasWHHItt955x3ee+89hg0bRm5uLvPnz2fLli0cc8wxtGvXjptuuok//vGPdOjQIa59ifac4IbUdfXVV8ecN2HCBACGDx/O559/zu7du+NqU20rVqzg+uuvB+CMM86gZ8+ebNy4EYCLLrqIY489lnbt2tG3b1+2bIk5lpNI8xfr6poUXnWjPvoGyMjIID8/n/z8fHJycpg/fz7jx49nypQplJaW0qNHD+65555DruGu7upp06ZNzefq6YMHDwKHXw5Ye9rdueSSSyguLj6sTStXruS1115j4cKFzJkzh9dff73e/fjnP/9Jnz59DinLzMyMu66jjjoqZt3R9iUzM5Nvvvmmpiyea9zremh95L9jRkZGzb+jSIuVwlCPRmf0cfrggw/48MMPa6bLysro2bNnTWh16dKFvXv3JvRl4ieffMLbb78NQHFxMeedd94h84cMGcLf//53Nm3aBMC+ffvYuHEje/fu5bPPPmPUqFE88sgjlJWV1butN954g8LCQm6++eZDymPV1bFjR/bs2RP3vixatAgInZEfe+yxHHvssWRnZ7N69WoAVq9ezccff1xv3cOHD2fBggUAbNy4kU8++YTTTz897naIyH+02DP6eC6HTKW9e/fywx/+kN27d5OZmcm3v/1tCgsL6dSpEzfffDM5OTlkZ2czaNCgBtfdp08f5s+fzw9+8AN69+7NLbfccsj8rKwsioqKmDBhAl9++SUA9913Hx07dmTMmDF88cUXuDsPP/xw1PoXLVrEihUr2LdvH7169eL5558/7Ix+z549UesaP348N998M48++mhcv8SOO+44hg4dyueff85TTz0FwJVXXsnTTz9Nbm4ugwYN4rTTTgOgc+fODBs2jP79+zNy5EhuvfXWmnqmTJnC5MmTycnJITMzk6KiokPO5EUkflbXn8jpkpeX57UfPLJhw4bDwikIysvLGT16NOvWrUt3UwIlqD8vIrGY2Sp3j3oTS71dN2bWw8yWmdkGM1tvZreHy483s1fM7MPw+3Ex1h9hZh+Y2SYzm57croiISEPF00d/EPixu/cBhgC3mllfYDrwmrv3Bl4LTx/CzDKAucBIoC8wIbyuhGVnZ+tsXkQaVb1B7+6fuvvq8Oc9wAagGzAGmB9ebD7w3SirnwNscveP3P0rYGF4PRERaSINuurGzLKBgcA/gBPd/VMI/TIAToiySjcg8m6WinBZtLoLzKzUzEqrqqoa0iwREalD3EFvZkcDzwM/cvfP410tSlnUb3/dvdDd89w9LysrK95miYhIPeIKejNrSyjkF7j7H8PF28ysa3h+V2B7lFUrgB4R092BysSbKyIiDRXPVTcGPAlscPeHIma9CEwMf54I/CnK6iVAbzPrZWZHAOPD67UoO3fuJDc3l9zcXE466SS6detWM53qESyrR3SMJSMjg9zcXPr168eAAQN46KGHau46LS0t5bbbbou5bnl5Oc8++2zM+ZWVlVx11VVAaPCwqVOnNqjtRUVFVFb+5/f4TTfdxHvvvdegOkSkEbh7nS/gPELdLe8CZeHXKKAzoattPgy/Hx9e/mRgScT6o4CNwGbg7vq25+6cffbZXtt77713WFk6zJw502fNmhXXsgcOHGhw/R9//LH369cv5vyjjjqq5vO2bdv8oosu8hkzZsRV97Jly/yyyy6LOq92W3//+9/7rbfeGle91S644AIvKSlp0DqNpbn8vIg0FaDUY2RqPFfdrHB3c/cz3T03/Fri7jvd/SJ3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)

从输出的图中可以看到采样值的分布与真实的分布之间的关系如下，采样集还是比较拟合对应分布的。

**M-H采样总结**

M-H采样完整解决了使用蒙特卡罗方法需要的任意概率分布样本集的问题，因此在实际生产环境得到了广泛的应用。

但是在大数据时代，M-H采样面临着两大难题：

- 1) 我们的数据特征非常的多，M-H采样由于接受率计算式$\frac{π(j)Q(j,i)}{π(i)Q(i,j)}$的存在，在高维时需要的计算时间非常的可观，算法效率很低。同时$α(i,j)$一般小于1，有时候辛苦计算出来却被拒绝了。能不能做到不拒绝转移呢？

- 2) 由于特征维度大，很多时候我们甚至很难求出目标的各特征维度联合分布，但是可以方便求出各个特征之间的条件概率分布。这时候我们能不能只有各维度之间条件概率分布的情况下方便的采样呢？

### 3.3.3 吉布斯抽样

> 相较于M-H采样，吉布斯抽样在多维分布且分布复杂但条件分布比较容易时表现得更好。总的来说是从时间复杂度的角度解决问题。

M-H 采样有两个缺点：一是需要计算接受率，在高维时计算量大。并且由于接受率的原因导致算法收敛时间变长。二是有些高维数据，特征的条件概率分布好求，但是特征的联合分布不好求。因此需要一个好的方法来改进 M-H 采样，这就是我们下面讲到的 Gibbs 采样。

M-H 采样由于接受率计算式![](https://www.zhihu.com/equation?tex=min%5C%7B%5Cfrac%7B%5Cpi%28j%29Q%28j%2Ci%29%7D%7B%5Cpi%28i%29Q%28i%2Cj%29%7D%2C1%5C%7D)的存在，在高维时需要的计算时间非常的可观，算法效率仍然很低。而且，很多时候我们甚至很难求出目标的各特征维度联合分布，但是可以方便求出各个特征之间的条件概率分布。所以我们希望对条件概率分布进行抽样，得到样本的序列。

用大白话说，吉布斯抽样解决的问题和 M-H 方法**解决的问题是一致的**，都是从给定一个已知的目标分布 ![](https://www.zhihu.com/equation?tex=p%28x%29) 中进行采样，并估计某个函数的期望值，区别只不过是此时， ![](https://www.zhihu.com/equation?tex=p%28x%29) 是一个多维的随机分布，![](https://www.zhihu.com/equation?tex=p%28x%29)的联合分布复杂，难以采样，但条件分布较容易，这样吉布斯抽样效果更好。

> 其基本做法是，**从联合概率分布定义条件概率分布，依次对条件概率分布进行抽样，得到样本的序列。可以证明这样的抽样过程是在一个马尔可夫链上的随机游走，每一个样本对应着马尔可夫链的状态，平稳分布就是目标的联合分布。**整体成为一个马尔可夫链蒙特卡罗法，燃烧期之后的样本就是联合分布的随机样本。

在前文中，我们讲到了细致平稳条件：如果非周期马尔科夫链的状态转移矩阵 $P$和概率分布$π(x)$对于所有的 $i,j$满足：

$$
π(i)P(i,j)=π(j)P(j,i)
$$
则称概率分布$π(x)$是状态转移矩阵 $P$的平稳分布。

在 M-H 采样中我们通过引入接受率使细致平稳条件满足。现在我们换一个思路。

![image-20201120181254658](C:\Users\LapTop-of-ChenWei\AppData\Roaming\Typora\typora-user-images\image-20201120181254658.png)

![image-20201120181316568](C:\Users\LapTop-of-ChenWei\AppData\Roaming\Typora\typora-user-images\image-20201120181316568.png)

接下来问题还是从二维分布 ![](https://www.zhihu.com/equation?tex=p%28x%29) 上进行采样，接下来这 3 个图代表以上所述状态转移的过程：

![](https://pic4.zhimg.com/v2-8bb8f4c2ac3b52ac061cd34dcd16ddbb_r.jpg)

如下图所示为二维的状态空间，ABC 代表 3 个不同的状态。

**对于第 1 个图：**

![](https://www.zhihu.com/equation?tex=%5Cpi%28x_%7B1%7D%5E%7B%281%29%7D%2Cx_%7B2%7D%5E%7B%281%29%7D%29%5Cpi%28x_%7B2%7D%5E%7B%282%29%7D%7Cx_%7B1%7D%5E%7B%281%29%7D%29%3D%5Cpi%28x_%7B1%7D%5E%7B%281%29%7D%29%5Cpi%28x_%7B2%7D%5E%7B%281%29%7D%7Cx_%7B1%7D%5E%7B%281%29%7D%29%5Cpi%28x_%7B2%7D%5E%7B%282%29%7D%7Cx_%7B1%7D%5E%7B%281%29%7D%29)

![](https://www.zhihu.com/equation?tex=%5Cpi%28x_%7B1%7D%5E%7B%281%29%7D%2Cx_%7B2%7D%5E%7B%282%29%7D%29%5Cpi%28x_%7B2%7D%5E%7B%281%29%7D%7Cx_%7B1%7D%5E%7B%281%29%7D%29%3D%5Cpi%28x_%7B1%7D%5E%7B%281%29%7D%29%5Cpi%28x_%7B2%7D%5E%7B%282%29%7D%7Cx_%7B1%7D%5E%7B%281%29%7D%29%5Cpi%28x_%7B2%7D%5E%7B%281%29%7D%7Cx_%7B1%7D%5E%7B%281%29%7D%29)

发现上面 2 个式子右端是一样的（故意写成这样的，方便后面推导）。所以：

![](https://www.zhihu.com/equation?tex=%5Cpi%28x_%7B1%7D%5E%7B%281%29%7D%2Cx_%7B2%7D%5E%7B%281%29%7D%29%5Cpi%28x_%7B2%7D%5E%7B%282%29%7D%7Cx_%7B1%7D%5E%7B%281%29%7D%29%3D%5Cpi%28x_%7B1%7D%5E%7B%281%29%7D%2Cx_%7B2%7D%5E%7B%282%29%7D%29%5Cpi%28x_%7B2%7D%5E%7B%281%29%7D%7Cx_%7B1%7D%5E%7B%281%29%7D%29)

可以写成：

![](https://www.zhihu.com/equation?tex=%5Cpi%28A%29%5Cpi%28x_%7B2%7D%5E%7B%282%29%7D%7Cx_%7B1%7D%5E%7B%281%29%7D%29%3D%5Cpi%28B%29%5Cpi%28x_%7B2%7D%5E%7B%281%29%7D%7Cx_%7B1%7D%5E%7B%281%29%7D%29)

这个式子好像和细致平衡方程有点像，此时 A 和 B 是在一条直线： ![](https://www.zhihu.com/equation?tex=x_1%3Dx_%7B1%7D%5E%7B%281%29%7D) 上的。A 点在上方，B 点在下方，可以看做是一维分布，所以上式可以**看作是**：

![](https://www.zhihu.com/equation?tex=%5Cpi%28%E4%B8%8A%29%5Cpi%28%E4%B8%8A%5Crightarrow%E4%B8%8B%29%3D%5Cpi%28%E4%B8%8B%29%5Cpi%28%E4%B8%8B%5Crightarrow%E4%B8%8A%29)

就是细致平衡方程，所以 ![](https://www.zhihu.com/equation?tex=%5Cpi%28x_%7B2%7D%7Cx_%7B1%7D%5E%7B%281%29%7D%29) 就是我们要找的马尔科夫链的状态转移概率。

**对于第 2 个图：**


![](https://www.zhihu.com/equation?tex=%5Cpi%28x_%7B1%7D%5E%7B%281%29%7D%2Cx_%7B2%7D%5E%7B%281%29%7D%29%5Cpi%28x_%7B1%7D%5E%7B%282%29%7D%7Cx_%7B2%7D%5E%7B%281%29%7D%29%3D%5Cpi%28x_%7B2%7D%5E%7B%281%29%7D%29%5Cpi%28x_%7B1%7D%5E%7B%281%29%7D%7Cx_%7B2%7D%5E%7B%281%29%7D%29%5Cpi%28x_%7B1%7D%5E%7B%282%29%7D%7Cx_%7B2%7D%5E%7B%281%29%7D%29)

![](https://www.zhihu.com/equation?tex=%5Cpi%28x_%7B1%7D%5E%7B%282%29%7D%2Cx_%7B2%7D%5E%7B%281%29%7D%29%5Cpi%28x_%7B1%7D%5E%7B%281%29%7D%7Cx_%7B2%7D%5E%7B%281%29%7D%29%3D%5Cpi%28x_%7B2%7D%5E%7B%281%29%7D%29%5Cpi%28x_%7B1%7D%5E%7B%282%29%7D%7Cx_%7B2%7D%5E%7B%281%29%7D%29%5Cpi%28x_%7B1%7D%5E%7B%281%29%7D%7Cx_%7B2%7D%5E%7B%281%29%7D%29)

发现上面 2 个式子右端是一样的。所以：

![](https://www.zhihu.com/equation?tex=%5Cpi%28x_%7B1%7D%5E%7B%281%29%7D%2Cx_%7B2%7D%5E%7B%281%29%7D%29%5Cpi%28x_%7B1%7D%5E%7B%282%29%7D%7Cx_%7B2%7D%5E%7B%281%29%7D%29%3D%5Cpi%28x_%7B1%7D%5E%7B%282%29%7D%2Cx_%7B2%7D%5E%7B%281%29%7D%29%5Cpi%28x_%7B1%7D%5E%7B%281%29%7D%7Cx_%7B2%7D%5E%7B%281%29%7D%29)

可以写成：

![](https://www.zhihu.com/equation?tex=%5Cpi%28A%29%5Cpi%28x_%7B1%7D%5E%7B%282%29%7D%7Cx_%7B2%7D%5E%7B%281%29%7D%29%3D%5Cpi%28C%29%5Cpi%28x_%7B1%7D%5E%7B%281%29%7D%7Cx_%7B2%7D%5E%7B%281%29%7D%29)

这个式子好像和细致平衡方程有点像，此时 A 和 C 是在一条直线： ![](https://www.zhihu.com/equation?tex=x_2%3Dx_%7B2%7D%5E%7B%281%29%7D) 上的。A 点在左侧，B 点在右侧，可以看做是一维分布，所以上式可以**看作是**：

![](https://www.zhihu.com/equation?tex=%5Cpi%28%E5%B7%A6%29%5Cpi%28%E5%B7%A6%5Crightarrow%E5%8F%B3%29%3D%5Cpi%28%E5%8F%B3%29%5Cpi%28%E5%8F%B3%5Crightarrow%E5%B7%A6%29)

就是细致平衡方程，所以 ![](https://www.zhihu.com/equation?tex=%5Cpi%28x_%7B1%7D%7Cx_%7B2%7D%5E%7B%281%29%7D%29) 就是我们要找的马尔科夫链的状态转移概率。

**对于第 3 个图：**

把上 2 图的结论整合在一起可以得到最终的结论：

![](https://www.zhihu.com/equation?tex=%5Cpi%28%E5%B7%A6%E4%B8%8B%29%5Cpi%28%E5%B7%A6%E4%B8%8B%5Crightarrow%E5%8F%B3%E4%B8%8A%29%3D%5Cpi%28%E5%8F%B3%E4%B8%8A%29%5Cpi%28%E5%8F%B3%E4%B8%8A%5Crightarrow%E5%B7%A6%E4%B8%8B%29)

就是细致平衡方程，也就是说，我们在不同的维度上分别将 ![](https://www.zhihu.com/equation?tex=%5Cpi%28x_%7B2%7D%7Cx_%7B1%7D%5E%7B%281%29%7D%29) 和 ![](https://www.zhihu.com/equation?tex=%5Cpi%28x_%7B1%7D%7Cx_%7B2%7D%5E%7B%281%29%7D%29) 作为马尔科夫链的状态转移概率，就能实现最终的效果，即：**马尔科夫链达到平稳后的一次随机游走等同于高维分布的一次采样**。这就是 Gibbs 采样。

![img](https://images2015.cnblogs.com/blog/1042406/201703/1042406-20170330161210195-1389960329.png)

**二维 Gibbs 采样的算法流程为：**

![image-20201120181600625](C:\Users\LapTop-of-ChenWei\AppData\Roaming\Typora\typora-user-images\image-20201120181600625.png)

![gibbs-algo-1](http://cos.name/wp-content/uploads/2013/01/gibbs-algo-1.jpg)

![image-20201120181641562](C:\Users\LapTop-of-ChenWei\AppData\Roaming\Typora\typora-user-images\image-20201120181641562.png)

![gibbs-algo-2](http://cos.name/wp-content/uploads/2013/01/gibbs-algo-2.jpg)

**二维 Gibbs 采样实例 python 实现**

假设我们要采样的是一个二维正态分布 ![](https://www.zhihu.com/equation?tex=N%28%5Cmu%2C%5CSigma%29) ，其中： ![](https://www.zhihu.com/equation?tex=%5Cmu%3D%28%5Cmu_%7B1%7D%2C%5Cmu_%7B2%7D%29%3D%285%2C-1%29) ， ![](https://www.zhihu.com/equation?tex=%5CSigma%3D%5Cbegin%7Bpmatrix%7D%5Csigma_%7B1%7D%5E%7B2%7D+%26%5Crho+%5Csigma_%7B1%7D%5Csigma_%7B2%7D+%5C%5C+%5Crho+%5Csigma_%7B1%7D%5Csigma_%7B2%7D+%26+%5Csigma_%7B2%7D%5E%7B2%7D%5Cend%7Bpmatrix%7D%3D%5Cbegin%7Bpmatrix%7D1+%261+%5C%5C+1+%26+4%5Cend%7Bpmatrix%7D) ;

首先需要求得：采样过程中的需要的状态转移条件分布：

![](https://www.zhihu.com/equation?tex=P%28x_%7B1%7D%7Cx_%7B2%7D%29%3DN%5Cleft%28%5Cmu_%7B1%7D%2B%5Crho+%5Csigma_%7B1%7D%2F%5Csigma_%7B2%7D%28x_%7B2%7D-%5Cmu_%7B2%7D%29%2C%281-%5Crho%5E%7B2%7D%29%5Csigma_%7B1%7D%5E%7B2%7D%5Cright%29%5C%5C)

![](https://www.zhihu.com/equation?tex=P%28x_%7B2%7D%7Cx_%7B1%7D%29%3DN%5Cleft%28%5Cmu_%7B2%7D%2B%5Crho+%5Csigma_%7B2%7D%2F%5Csigma_%7B1%7D%28x_%7B1%7D-%5Cmu_%7B1%7D%29%2C%281-%5Crho%5E%7B2%7D%29%5Csigma_%7B2%7D%5E%7B2%7D%5Cright%29%5C%5C)

证：

![](https://www.zhihu.com/equation?tex=f%28x_1%2Cx_2%29%3D%5Cfrac%7B1%7D%7B2%5Cpi%5Csigma_1%5Csigma_2%5Csqrt%7B1-%5Crho%5E2%7D%7Dexp%5Cleft%5C%7B+-%5Cfrac%7B1%7D%7B2%281-%5Crho%5E2%29%7D%5Cleft%5B+%28%5Cfrac%7Bx_1-%5Cmu_1%7D%7B%5Csigma_1%7D%29%5E2-2r%28%5Cfrac%7Bx_1-%5Cmu_1%7D%7B%5Csigma_1%7D%29%28%5Cfrac%7Bx_2-%5Cmu_2%7D%7B%5Csigma_2%7D%29%2B%28%5Cfrac%7Bx_2-%5Cmu_2%7D%7B%5Csigma_2%7D%29%5E2+%5Cright%5D+%5Cright%5C%7D)

![](https://www.zhihu.com/equation?tex=f%28x_2%29%3D%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D%5Csigma_2%7Dexp%5Cleft%5C%7B+-%5Cfrac%7B%28x-%5Cmu_2%29%5E2%7D%7B2%5Csigma_2%5E2%7D+%5Cright%5C%7D)

所以：

![](https://www.zhihu.com/equation?tex=f%28x_1%7Cx_2%29%3D%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D%5Csigma_1%5Csqrt%7B1-%5Crho%5E2%7D%7Dexp%5Cleft%5C%7B+-%5Cfrac%7B1%7D%7B2%281-%5Crho%5E2%29%5Csigma_1%5E2%7D%5Cleft%5B+x_1-%5Cmu_1-%5Cfrac%7B%5Csigma_1%7D%7B%5Csigma_2%7D%5Crho%28x_2-%5Cmu_2%29%5Cright%5D%5E2+%5Cright%5C%7D)

得证。

```python
import random
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # 绘制三维图
from scipy.stats import multivariate_normal  # 多维正态

def p_x2givenx1(x1, m1, m2, s1, s2):
    return random.normalvariate(m2 + rho * s2 / s1 * (x1 - m1), math.sqrt(1 - rho**2)*s2)

def p_x1givenx2(x2, m1, m2, s1, s2):
    return random.normalvariate(m1 + rho * s1 / s2 * (x2 - m2), math.sqrt(1 - rho**2)*s1)

N = 5000
K = 20
x1_list = []
x2_list = []
z_list = []
m1 = 5
m2 = -1
s1 = 1
s2 = 2

rho = 0.5
x2 = m2   # 初值为均值

# 用来计算二元正态的概率密度
samplesource = multivariate_normal(mean=[m1,m2], cov=[[s1**2,rho*s1*s2],[rho*s1*s2,s2**2]])

for i in range(N):
    for j in range(K):
        x1 = p_x1givenx2(x2, m1, m2, s1, s2) #x2给定得到x1的采样
        x2 = p_x2givenx1(x1, m1, m2, s1, s2) #x1给定得到x2的采样
        z = samplesource.pdf([x1, x2])
        x1_list.append(x1)
        x2_list.append(x2)
        z_list.append(z)

num_bins = 50
plt.scatter(x1_list, norm.pdf(x1_list, loc=m1, scale=s1),label='Target Distribution x1', c= 'green')
plt.scatter(x2_list, norm.pdf(x2_list, loc=m2, scale=s2),label='Target Distribution x2', c= 'orange')
plt.hist(x1_list, num_bins, density=1, facecolor='Cyan', alpha=0.5,label='x1')
plt.hist(x2_list, num_bins, density=1, facecolor='magenta', alpha=0.5,label='x2')
plt.title('Histogram')
plt.legend()
plt.show() 
```

![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXwAAAEICAYAAABcVE8dAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3de5gU1bnv8e/LABtQonFAoiBClMQgOIjDxRAV4g2ikXjLVjFuop7Z7IRozMk+kEMiQ9zu6Ikx0UcMGwlB9wFJYqIhCUa8BDUnXmZIBhFFRS46wQBiVBCIXN7zR9cMTU/3dHVPX6q7f5/nmWe6qtaqfrt65u3Vq1atMndHRETKX6diByAiIoWhhC8iUiGU8EVEKoQSvohIhVDCFxGpEEr4IiIVQglfypKZrTazscWOQyRKlPClJJnZBjM7K2HdZDP7I4C7n+juy9PsY4CZuZl1zmOoIpGhhC+SJ/ogkahRwpeyFP8NwMxGmlmjmb1vZpvN7Pag2FPB73fNbIeZnWpmnczs22a20cy2mNl9ZnZY3H6vCrZtM7PvJDxPvZk9YGb/18zeByYHz/2Mmb1rZm+Z2V1m1jVuf25mXzGz18xsu5ndZGbHBXXeN7Ofx5cX6QglfKkEdwB3uPtHgOOAnwfrTw9+H+7uh7r7M8Dk4Gcc8HHgUOAuADMbDNwNTAKOAg4D+iY810TgAeBwYCGwD7gB6AWcCpwJfCWhznjgFGA08L+AucFzHAMMAS7vwGsXaaWEL6XsoaDl/K6ZvUssGSezBzjezHq5+w53f7adfU4Cbnf3de6+A/gWcFnQPXMJ8Bt3/6O7fwjcCCRORvWMuz/k7vvdfZe7r3D3Z919r7tvAP4LOCOhzq3u/r67rwZeBJYFz/8e8DBwcvhDIpKaEr6Usi+4++EtP7RtObe4BvgEsMbMGszs/Hb2eTSwMW55I9AZ6BNse7Nlg7vvBLYl1H8zfsHMPmFmvzWzvwXdPP9JrLUfb3Pc411Jlg9tJ16R0JTwpey5+2vufjlwJHAr8ICZHULb1jnAJuDYuOX+wF5iSfgtoF/LBjPrDlQnPl3C8o+BNcCgoEvpfwOW/asRyZ4SvpQ9M7vSzHq7+37g3WD1PmArsJ9YX32L+4EbzGygmR1KrEX+M3ffS6xv/vNm9ungROos0ifvnsD7wA4zOwH4t5y9MJEMKeFLJRgPrDazHcRO4F7m7ruDLpmbgf8XnAcYDcwH/pvYCJ71wG7gawBBH/vXgMXEWvvbgS3AP9p57m8CVwRl7wF+lvuXJxKO6QYoItkJvgG8S6y7Zn2x4xFJRy18kQyY2efNrEdwDuA2YBWwobhRiYSjhC+SmYnETuxuAgYR6x7S12QpCerSERGpEGrhi4hUiEhO7tSrVy8fMGBAscMQESkZK1aseNvde7dXJpIJf8CAATQ2NhY7DBGRkmFmG9OVUZeOiEiFUMIXEakQSvgiIhUikn34yezZs4fm5mZ2795d7FAkgrp160a/fv3o0qVLsUMRiaySSfjNzc307NmTAQMGYKbJBuUAd2fbtm00NzczcODAYocjElkl06Wze/duqqurleylDTOjurpa3/5E0giV8M1svJm9YmZrzWx6O+VGmNk+M7sk07oh4+hIdSlj+tsQSS9tl46ZVQGzgbOBZqDBzJa4+0tJyt0KPJJpXREpXzYr9Yexz9TULoUUpg9/JLDW3dcBmNliYhNIJSbtrwG/BEZkUTdj9R3dQZ73JyLtJ/uW7Ur6hROmS6cvB9+nszlY18rM+gIXAnMyrRu3jzozazSzxq1bt4YIS0SiLGmyP2PmgZ/2yklehEn4yd6NxI/kHwHT3H1fFnVjK93nunutu9f27t3udBBF0dDQwEknncTu3bv54IMPOPHEE3nxxReLHZZIJGWaxJX0CyNMl04zcEzccj9ic4HHqwUWByfOegGfM7O9IeuWhBEjRnDBBRfw7W9/m127dnHllVcyZMiQYoclIhJamITfAAwys4HAX4HLiN2js5W7tw5+NrMFwG/d/SEz65yubim58cYbGTFiBN26dePOO+8sdjgiIhlJ26Xj7nuBqcRG37wM/NzdV5vZFDObkk3djoddHO+88w47duxg+/btGvMtkkK23TPq1sm/UFfauvtSYGnCusQTtC3rJ6erW6rq6uq46aabWL9+PdOmTeOuu+4qdkgi0Rd3gjbp+idnFS6WClcyUyskqi/w891333107tyZK664gn379vHpT3+aJ554gs9+9rMFjkREJDslm/AL7aqrruKqq64CoKqqiueee67IEYlET0e7ZTQuP7+U8EUkP1J15UjRlMzkaSJS+maOrS92CBVNCV9EpEIo4YtITuRqWKWGZ+aPEr6IFERLd466dYpHCV9EpEKU7iid+ojvT0QkYko34YtI9GgoZqSpSyek73znO9xxxx2tyzNmzNAEaiKBdCdaE/vt0/Xj68Rtfijhh3TNNddw7733ArB//34WL17MpEmTihyViEh46tIJacCAAVRXV/OXv/yFzZs3c/LJJ1NdXV3ssEREQlPCz8C1117LggUL+Nvf/sbVV19d7HBEykN8v79mzswrdelk4MILL+T3v/89DQ0NnHvuucUOR0QkI6Xbwq8v/FN27dqVcePGcfjhh1NVVVX4AEQiKNMTtvHrZy1Pvq1lv5o5M7dKN+EXwf79+3n22Wf5xS9+UexQREQyFqpLx8zGm9krZrbWzKYn2T7RzF4wsyYzazSzz8Rt22Bmq1q25TL4QnrppZc4/vjjOfPMMxk0aFCxwxERyVjaFr6ZVQGzgbOBZqDBzJa4+0txxR4Hlri7m9lJwM+BE+K2j3P3t3MYd8ENHjyYdevWFTsMEZGshWnhjwTWuvs6d/8QWAxMjC/g7jvcvaWz7RBAHW8iIhETJuH3Bd6MW24O1h3EzC40szXA74D4MYsOLDOzFWZWl+pJzKwu6A5q3Lp1a7joRSTS0l1Rq5kzCytMwk92Cr5NC97dH3T3E4AvADfFbRrj7sOBCcBXzez0ZE/i7nPdvdbda3v37h0iLBEptnxPgaApFnIrTMJvBo6JW+4HbEpV2N2fAo4zs17B8qbg9xbgQWJdRCIiUmBhEn4DMMjMBppZV+AyYEl8ATM73swseDwc6ApsM7NDzKxnsP4Q4BzgxVy+gELYtm0bw4YNY9iwYXzsYx+jb9++rcsffvhhTp/r3Xff5e677065vaqqimHDhnHiiSdSU1PD7bffzv79+wFobGzkuuuuS1l3w4YNLFq0KOX2TZs2cckllwCwYMECpk6dmlHsCxYsYNOmA22Ba6+9lpdeeqmdGiJSSGlH6bj7XjObCjwCVAHz3X21mU0Jts8BLgauMrM9wC7gn4MRO32AB4PPgs7AInf/fZ5eS95UV1fT1NQEQH19PYceeijf/OY309bbu3cvnTtndqlDS8L/yle+knR79+7dW2PZsmULV1xxBe+99x6zZs2itraW2tralPtuSfhXXHFF0liPPvpoHnjggYzijbdgwQKGDBnC0UcfDcC8efOy3peI5F6ocfjuvtTdP+Hux7n7zcG6OUGyx91vdfcT3X2Yu5/q7n8M1q9z95rg58SWuoWwcNVCBvxoAJ1mdWLAjwawcNXCnO7/nnvuYcSIEdTU1HDxxRezc+dOACZPnsw3vvENxo0bx7Rp03j99dcZPXo0I0aM4MYbb+TQQw9t3cf3v/99RowYwUknncTMmbH5RKZPn87rr7/OsGHD+Pd///d2YzjyyCOZO3cud911F+7O8uXLOf/88wF48sknW7+FnHzyyWzfvp3p06fz9NNPM2zYMH74wx+yYMECLr30Uj7/+c9zzjnnsGHDBoYMGdK6/zfffJPx48fzyU9+klmzYnOcJJa57bbbqK+v54EHHqCxsZFJkyYxbNgwdu3axdixY2lsjF16cf/99zN06FCGDBnCtGnTWusfeuihzJgxg5qaGkaPHs3mzZvbvM7rrruO7373uwA88sgjnH766a3fakQkvLKcS2fhqoXU/aaOje9txHE2vreRut/U5TTpX3TRRTQ0NLBy5Uo+9alP8ZOf/KR126uvvspjjz3GD37wA66//nquv/56GhoaWlu+AMuWLeO1117j+eefp6mpiRUrVvDUU09xyy23cNxxx9HU1MT3v//9tHF8/OMfZ//+/WzZsuWg9bfddhuzZ8+mqamJp59+mu7du3PLLbdw2mmn0dTUxA033ADAM888w7333ssTTzzRZt/PP/88CxcupKmpiV/84hetyTuZSy65hNra2tby3bt3b922adMmpk2bxhNPPEFTUxMNDQ089NBDAHzwwQeMHj2alStXcvrpp3PPPfe02fctt9zCz372M/7whz9w3XXX8dOf/pROncryT7c0nTHzwE+csCNwNFKncMryv2bG4zPYuWfnQet27tnJjMdn5Ow5XnzxRU477TSGDh3KwoULWb16deu2Sy+9tHWunWeeeYZLL70U4KCulGXLlrFs2TJOPvlkhg8fzpo1a3jttdeyiuXAJRAHjBkzhm984xvceeedvPvuuym7ls4++2yOOOKIlNuqq6vp3r07F110EX/84x+ziq+hoYGxY8fSu3dvOnfuzKRJk3jqqaeA2PxELd9KTjnlFDZs2NCmfo8ePbjnnns4++yzmTp1Kscdd1xWcYhUurKcS+eN997IaH02Jk+ezEMPPURNTQ0LFixg+fLlrdsOOeSQtPXdnW9961v867/+60HrkyW89qxbt46qqiqOPPJIXn755db106dP57zzzmPp0qWMHj2axx57LGn99mINzr0ctNy5c+eDulN2796dNsZkH0gtunTp0vo8VVVV7N27N2m5VatWUV1dfdBJYSmuvAyZTDJVsiZRy52ybOH3P6x/RuuzsX37do466ij27NnDwoWpu4pGjx7NL3/5SwAWL17cuv7cc89l/vz57NixA4C//vWvbNmyhZ49e7J9+/ZQMWzdupUpU6YwderUNsn59ddfZ+jQoUybNo3a2lrWrFmT0b4BHn30Ud555x127drFQw89xJgxY+jTpw9btmxh27Zt/OMf/+C3v/1ta/lU+x81ahRPPvkkb7/9Nvv27eP+++/njDPOCB3Hxo0b+cEPfsBf/vIXHn74YZ577rnQdUXkgLJM+DefeTM9uvQ4aF2PLj24+czcnTO+6aabGDVqFGeffTYnnHBCynI/+tGPuP322xk5ciRvvfUWhx12GADnnHMOV1xxBaeeeipDhw7lkksuYfv27VRXVzNmzBiGDBmS9KTtrl27WodlnnXWWZxzzjmtJ3wTn3fIkCHU1NTQvXt3JkyYwEknnUTnzp2pqanhhz/8YdrX+JnPfIYvfelLDBs2jIsvvpja2lq6dOnCjTfeyKhRozj//PMPeu2TJ09mypQprSdtWxx11FF873vfY9y4cdTU1DB8+HAmTpyY7CnbcHeuueYabrvtNo4++mh+8pOfcO2114b6ZiEiB7P2vm4XS21trSeeIHz55Zf51Kc+FXofC1ctZMbjM3jjvTfof1h/bj7zZiYNLfw9aHfu3En37t0xMxYvXsz999/Pr3/964LHUQky/RuRjmnt0jmjbYMDMjsZm3Re/Li7X6lLJz0zW+HuqcdlU6Z9+ACThk4qSoJPtGLFCqZOnYq7c/jhhzN//vxihyQiFapsE35UnHbaaaxcubLYYYgUVKZDLdPd/Upyoyz78EWkvOT6wslKpYQvIhkr9CyWV/7qyoI+X7lSwhcRqRBK+CIiFUIJPwRNjxyOpkcWiTYl/BBapkduampiypQp3HDDDa3LXbt2TVkv1TQB7UmX8FumR169ejWPPvooS5cubZ3Jsra2ljvvvDNl3fYSfq6mR45P+PPmzWPw4MFZ708qiyZRy7/yTfjrF8JDA2BRp9jv9ZoeuRSnR96/fz+DBg2i5T7H+/fv5/jjj+ftt9/O7A2TglHijq7yTPjrF8LzdbBzI+Cx38/X5TTpa3rkg+VreuROnTpx5ZVXts5X9Nhjj1FTU0OvXr3SHhsROVh5JvyVM2DfwdMjs29nbH2OaHrk8Do6PfLVV1/NfffdB8D8+fP58pe/nFUckhvFurG4bmjecaESvpmNN7NXzGytmU1Psn2imb1gZk1m1mhmnwlbNy92ppgGOdX6LEyePJm77rqLVatWMXPmzIMm88pkeuSWcwFr167lmmuuyTiO+OmR402fPp158+axa9cuRo8ezZo1a5LWL4XpkY855hj69OnDE088wXPPPceECRPSPqeItJU24ZtZFTAbmAAMBi43s8QzcY8DNe4+DLgamJdB3dzrkWIa5FTrs6DpkQs3PTLERvxceeWVfPGLX2z99iRFluQuVxJtYVr4I4G1wf1pPwQWAwfNbevuO/xAM+4QwMPWzYuam6Hq4OmRqeoRW58jmh65MNMjt7jgggvYsWOHunNEOiDt9Mhmdgkw3t2vDZa/BIxy96kJ5S4EvgccCZzn7s+ErRtsqwPqAPr373/Kxo0bD9qe8dS36xfG+ux3vhFr2dfcDAM1PXKpamxs5IYbbuDpp59OWUbTIxdGLqdFTnTQBGpx0yO30DTJqeVqeuRkZ0raHHV3fxB40MxOB24CzgpbN6g/F5gLsfnwQ8TVvoGTipLgE2l65I675ZZb+PGPf9xu15lEQ0eHZGrWzPwKk/CbgWPilvsBKW8s6u5PmdlxZtYr07rlSNMjd9z06dOZPr0w5/tFylmYPvwGYJCZDTSzrsBlwJL4AmZ2vAVnDc1sONAV2BambiaieHcuiQb9bVQGDc3smLQtfHffa2ZTgUeAKmC+u682synB9jnAxcBVZrYH2AX8c3ASN2ndbALt1q0b27Zto7q6us2IFKls7s62bdvo1q1bsUMpe0q4pS3UHa/cfSmwNGHdnLjHtwK3hq2bjX79+tHc3Nx6ib1IvG7dutGvX79ihyESaSVzi8MuXbowcODAYochIoXSMgooyWgdyU55Tq0gIiJtKOGLSKRots38UcIXkZxQoo4+JXwRkQqhhC8ioURlSGZU4ihFSvgiIhVCCV9EpEIo4YuIVAglfBGRCqGELyIdlushmRrimR9K+CIiFUIJX0SkQijhi0haURv7HrV4SoUSvohIhVDCFxGpECUzH76IREDLHPVSkkK18M1svJm9YmZrzazN3aTNbJKZvRD8/MnMauK2bTCzVWbWZGaNuQxeRETCS5vwzawKmA1MAAYDl5vZ4IRi64Ez3P0k4CZgbsL2ce4+zN1rcxCziERIvsbMayx+7oVp4Y8E1rr7Onf/EFgMTIwv4O5/cve/B4vPArq5qIhIxITpw+8LvBm33AyMaqf8NcDDccsOLDMzB/7L3RNb/wCYWR1QB9C/f/8QYYlIRYg/b6D723ZImBZ+sgGvnrSg2ThiCX9a3Oox7j6cWJfQV83s9GR13X2uu9e6e23v3r1DhCUihRDVMe9RjSvKwiT8ZuCYuOV+wKbEQmZ2EjAPmOju21rWu/um4PcW4EFiXUQiIlJgYRJ+AzDIzAaaWVfgMmBJfAEz6w/8CviSu78at/4QM+vZ8hg4B3gxV8GLiEh4afvw3X2vmU0FHgGqgPnuvtrMpgTb5wA3AtXA3WYGsDcYkdMHeDBY1xlY5O6/z8srERGRdoW68MrdlwJLE9bNiXt8LXBtknrrgJrE9SIiUniaWkFEspbvsfIai59bSvgiIhVCCV9EpEIo4YtISlEf6x71+KJGCV9EpEIo4YuIVAglfBGRCqGELyJSIZTwRSQrhRojr7H4uaOELyJSIZTwRUQqhBK+iCRVKmPcSyXOKFDCFxGpEEr4IiIVQglfRKRChJoPX0QkEnRD8w4J1cI3s/Fm9oqZrTWz6Um2TzKzF4KfP5lZTdi6IhJxZ8w88BMo9Nh4jcXPjbQtfDOrAmYDZxO7oXmDmS1x95fiiq0HznD3v5vZBGAuMCpkXZH8WpThKI4rPD9xiBRZmC6dkcDa4HaFmNliYCLQmrTd/U9x5Z8F+oWtK5I3mSb6xHpK/FJmwnTp9AXejFtuDtalcg3wcJZ1RTpukWWf7POxnxJUamPbSy3eYgnTwk92JJM2fcxsHLGE/5ks6tYBdQD9+/cPEZZIEokJ+pcz25a5OMOTfYtMrX0pC2ESfjNwTNxyP2BTYiEzOwmYB0xw922Z1AVw97nE+v6pra3Vf5dkLmxrPNmHALT/QaCkL2UgTJdOAzDIzAaaWVfgMmBJfAEz6w/8CviSu7+aSV2RnChE10uFdu9I+Ujbwnf3vWY2FXgEqALmu/tqM5sSbJ8D3AhUA3ebGcBed69NVTdPr0Uq1SJL3WrPRJiWv1r6UsJCXXjl7kuBpQnr5sQ9vha4NmxdkZzJttU9tD72e1V9ds9ZwUm/WGPiZ46tZ9by4jx3udCVtlK6skn2LYk+cTnTxF/hSV9Kk+bSkdKUabIfWt822WeyPRcxSF71+I8exQ4h8tTCl/KXSSIfWt+2tZ+LoZ0lpFTHtO/at6vYIUSeWvhSejJpWWfaas+0jlr5UkKU8KW05DvZZ1NXSV9KhBK+lKeOJPtc7kMkQpTwpXSEbUnnMlGH3Zda+QWhaZI7RidtpTS0JNRcXGCVL+U2VPOMtsc6Ugm3JT7dCCU0tfClvOSjG0ZdO1ImlPAl+orRlZPtvku8a6dUh2S2KPX4800JX6ItSgm030XhykUpZpE4SvhSHgrR7fLRk/L/HCJ5pJO2El1R6MpJ9lyr6tPPrFluJ3ClLKiFLyJSIZTwJZqi2LrP9DnLrC8/KkMyoxJHKVLCl9JVzOGSGqopJUgJX6KnnFrGJfRaymVIY7m8jnwIddLWzMYDdxC7TeE8d78lYfsJwE+B4cAMd78tbtsGYDuwj+DWh7kJXcpS2Ctqo9DCTjaVcjI6gSsRkTbhm1kVMBs4G2gGGsxsibu/FFfsHeA64AspdjPO3d/uaLAiIpK9MF06I4G17r7O3T8EFgMT4wu4+xZ3bwD25CFGqRRRPlGbSoWewJXSFCbh9wXejFtuDtaF5cAyM1thZnWpCplZnZk1mlnj1q1bM9i9VBTrVuwI2orSB5BIO8Ik/GRNk0w6JMe4+3BgAvBVMzs9WSF3n+vute5e27t37wx2L2UhbAt4yPT8xpFPJdzKj9pQyKjFUyrCnLRtBo6JW+4HbAr7BO6+Kfi9xcweJNZF9FQmQYoA0W5Jh70CV6SIwrTwG4BBZjbQzLoClwFLwuzczA4xs54tj4FzgBezDVbKVAm3fDMW0ddabkMZy+315EraFr677zWzqcAjxIZlznf31WY2Jdg+x8w+BjQCHwH2m9nXgcFAL+BBM2t5rkXu/vv8vBQpa1Fu3bcIO0wzypLc9ETKR6hx+O6+FFiasG5O3OO/EevqSfQ+UNORAKXMRbTFm1caly9FotkyJfpKoXXfohxa+aUm/luJbnfYLiV8KZ5SuE9tvqiVL0WguXQk2kqpdd+iFGNuR1SHQEY1rihTwpfiqMS++0Q6BlJgSvgSXaXcUi6h2Mt1CGO5vq6OUMKXwlPL9gAdCykgnbSVaCqhFnJK8SN2dAWuRIBa+FJYatG2pWMiBaKEL9FTDq37FuX0WiKok1UVO4SSooQvhaOWbGoRPTZRH/r4nTO+U+wQSooSvkRLObaII/yayn0kS7m/vkwp4UthRLQFGyk6RpJnGqUjhVMKNybPF82xIxGgFr7kn1qu4elYSR4p4Us0lHPrvkUlvEaJNCV8yS+1WDNX6GN2xswDP3GiPkKnRanEGQVK+FJ8ldTyjdBrrZQRLJXyOsMIlfDNbLyZvWJma81sepLtJ5jZM2b2DzP7ZiZ1pYypdZ89HTvJg7QJ38yqgNnABGL3qb3czAYnFHsHuA64LYu6Uski1OItmEp8zYWSontKYsK08EcCa919nbt/CCwGJsYXcPct7t4A7Mm0rpQptVA7TsdQcizMOPy+wJtxy83AqJD7D13XzOqAOoD+/fuH3L2UtEpu6baMy092bYJm0JQ8CdPCT9bMCHszztB13X2uu9e6e23v3r1D7l4iSS3T3MnTsUx3IrPURr6ki1cnbmPCJPxm4Ji45X7AppD770hdKWeV3LpvoWMgBRYm4TcAg8xsoJl1BS4DloTcf0fqSilS6z73dEwlR9L24bv7XjObCjwCVAHz3X21mU0Jts8xs48BjcBHgP1m9nVgsLu/n6xuvl6MREQlz5mTKc2xIwUUavI0d18KLE1YNyfu8d+IddeEqitlSi3R/FlkcEXYU2ciyelKW8mNsMlerfu2wh6THH2gnjj7xHa3l9oJ2xY6cZueEr5IhXnp7ZeKHYIUiRK+dJxa9x1X4Fa+VCbdAEUkalKd9NYFWdJBauFLx6h1nztq5eeW5tRpQwlf8k/JPrw8H6tyu8I2kU7ctk9dOpI9tTSLJxfDNNX6rThq4Ut2fvbRcOXUus+cunYkT9TCl+zsezf9FbUiEilq4UvmdKI2/9TKz5tK7sdXwhepEOV+wrZFubyOfFDCl8yodV84auVLjinhS3hK9oWnpC85pJO2IqVCV+DmjM0yfGblzT6qFr6Eo9Z98eSglZ+u/37gRz+eQUDRp3785JTwJT11F5SOLN+rq2quynEgEkXq0pHcUes+f3J1Z6xKvLo2/jU/WdndX6Fa+GY23sxeMbO1ZjY9yXYzszuD7S+Y2fC4bRvMbJWZNZlZYy6DlwJQV050ZNm1U8njzttTicclbQvfzKqA2cDZQDPQYGZL3D3+LgoTgEHBzyjgx8HvFuPc/e2cRS2FEZ84dFVtaclgrp3ao0fkOZjimDm2nlnL64sdRqSEaeGPBNa6+zp3/xBYDExMKDMRuM9jngUON7OjchyrRJVa94WTh2N93ifOy/k+JZrCJPy+wJtxy83BurBlHFhmZivMrC7Vk5hZnZk1mlnj1q1bQ4QleaWunOjKoGvnrPvOymsopa7SunXCnLRNdkQSvyu2V2aMu28ysyOBR81sjbs/1aaw+1xgLkBtbW3lDZCNEo3KKS3tjM9f1ulxqgobjURYmBZ+M3BM3HI/YFPYMu7e8nsL8CCxLiKJqkySvVr3xRPy2JvBvuNSby/38erl/voyFSbhNwCDzGygmXUFLgOWJJRZAlwVjK5waiwAAAuBSURBVNYZDbzn7m+Z2SFm1hPAzA4BzgFezGH8kktK9qUlxHtglj7pV7pK6tZJ26Xj7nvNbCrwCFAFzHf31WY2Jdg+B1gKfA5YC+wEvhxU7wM8aGYtz7XI3X+f81chHZdRN07l/INEnnUD391+keDt2ncc6t6pcKEuvHL3pcSSevy6OXGPHfhqknrrgJoOxiiFFGb45VAN0YyMIdPTXpBVH7xf7sA7B9ZXSneHhmceoKkVRF05pS6D9+TGI/IXRkk4Y+aBnziV0q2jhF/plOzLQ8j+fFDSr2RK+JVMyb68BO+RtzOouSXpz6zOfzhREqb7qsd/9Mh/IEWmydMq0aIqYH/48kr2pWNoPbxQH3scnI8Z+/TY1s3Lv7K8Nem39v3r/QVg175dxQ4h79TCrzSLjIySvZScvbTfym8jF7NwSklQwq8k2VxBq9ZfSalfXs/NwUicPxw5luVHjg1XsQKSfphunXI/easunUqRmOxDDb+sz0sokh/3rbyv9fGsd+CMPrH5TUKnsFX1lfWet4zUSZgjv5xvf6gWfrlbZGrZV4h1f1930PKTQZd0RqlrVX1Zt/Yr5dqDVJTwy1m2k6Ap2Zec+hQXFmWV9KGsk34Y5dq1oy6dctSR2S6V7MvOk7vgtO6x1t3Yu8cmLbP8K8vbrizTUTxhr7w9676zeOyqx/IfUAEp4ZcTJfqKdFDrfsDYpGWe3gVjB4yFzcszf4IyTfzpPL7+8WKHkHPmGY3fKoza2lpvbNTtb0NLl+jTnaCtsH/kctKmKydFwh8bvz6DpJ+05V8mfy9JW/lJbnJeKidwzWyFu9e2V0Yt/FKVq5uUlMk/byVK1W+fVp+x2bX0W8T371fA3085jdpRwi8lHW3JJ6qAf9ZyFLZV32Jssu0dTfotSjj5ZzKLZstJ3FJP/Er4UZavWw2W2D+mxNz81M3s2b8nozpH9zw69cY+Y2O/c5H4oe3InhL4O2uT9ONn0UzSvWOzjO5V3dn57Z35Dy4PlPCL7bGzYEuBTg5V9YTB/7MwzyU58cLmF/jVy7/Kuv4nqj+RvlCfsfD+q7Ar8c6lyUf1JO3XTybV0M5+F8FHTwq3jwjatW9Xa4v/32r/jbvPu7vIEYWnk7a5UsjEnWnXDZREa6uSzX5+Nlt3bs2sUjZdOel0oLUf+oMgKwb9LszbB0XYE7hhnTnwzIIP6Qxz0jZUwjez8cAdxO6QNs/db0nYbsH2zxG7xeFkd/9zmLrJZJXw1y+ElTNg5xvQoz/U3Bxbn7hu4KS25Tv1gP0fxL8i2r9UJd32HMkmsSdSoo+c3736OxrfaiSrxlbIJJ5Vso+XReLPb8LPwhEjoO95scd/fwE2PwF73oMuh0Gfz7b58AjVn5/wIXD5ofCfvaB/Z3hjL/zvt+H+Hel3U2VV1J1S1/rtYOGqhcx4fAZvvPcG/Q/rz81n3sykoZPCvMpWOUn4ZlYFvAqcDTQTu6n55e7+UlyZzwFfI5bwRwF3uPuoMHWTyTjhr18Iz9fBvrh+NesSm/x7/4cH1lX1gJFzY48TyxdbLpJ7C3XdRNbvXv0dDZsaMquUYfLucLKP9/eV8OHfO7SLon4QHDECehwDf/0NeNz5D+sCfT+fXdIPXL5iFvf0gUPi5iv4YD/8j83hkj7EuoTG9B9D3W/q2LnnQD7q0aUHcz8/N6Okn6uEfypQ7+7nBsvfAnD378WV+S9gubvfHyy/AowFBqSrm0zGCf+hAbBzY7iyPY6N/Q5bPlu5TOBhqTUfed998rvs93amp+5Asu5knTj92NOzrp9Wrk7uZiH7D41O0OUjsOfdtpu6HA4nfL3N6rBJ//rD4fCEu8LXr5rFhj0wcEO46Kqsin4f6cfG99rmo2MPO5YNXw+5I3I3Dr8v8GbccjOxVny6Mn1D1m0Jtg6oCxZ3BB8aoZwykFO2bofePcOUznOib3Xgq1/42DLjjv95A39uXfGrrPocewFv5yqmHCu/2I7ilHa3v5plv/FbrIhbyvtxGz6Ak80yn4sr6/+FR7KoE8KK9TekPm5p3qs/dmu77rctD3aHe/597GNjipy0kY3YDdYSX5j39Nh0zxcm4ScbG5j4tSBVmTB1Yyvd5wJzQ8STlJk1btza/qdbsUQ9tnStgmJRbNmJemz6X8hcrmILk/CbgWPilvsBieO3UpXpGqKuiIgUQJivZA3AIDMbaGZdgcuAJQlllgBXWcxo4D13fytkXRERKYC0LXx332tmU4n1olUB8919tZlNCbbPAZYSG6GzltiwzC+3Vzcvr6QD3UEFoNiyo9iyo9iyU/axRfLCKxERyT3d8UpEpEIo4YuIVIiSSvhmdqmZrTaz/WZWm7DtW2a21sxeMbNzU9Q/wsweNbPXgt8fzVOcPzOzpuBng5k1pSi3wcxWBeUKMnmQmdWb2V/j4vtcinLjg2O51symFyi275vZGjN7wcweNLPDU5Qr2HFLdxyCgQp3BttfMLPh+Ywn7nmPMbM/mNnLwf/E9UnKjDWz9+Le6xsLEVvw3O2+R0U8bp+MOx5NZva+mX09oUzBjpuZzTezLWb2Yty6UHkqq/9Rdy+ZH+BTwCeB5UBt3PrBwErgn4CBwOtAVZL6/weYHjyeDtxagJh/ANyYYtsGoFeBj2E98M00ZaqCY/hxYkNrVwKDCxDbOUDn4PGtqd6fQh23MMeB2GCFh4ldczIaeK5A7+NRwPDgcU9iU5gkxjYW+G0h/77CvkfFOm5J3t+/AccW67gBpwPDgRfj1qXNU9n+j5ZUC9/dX3b3ZFfgTgQWu/s/3H09sdFCI1OUuzd4fC/whfxEGmNmBnwRuD+fz5MHI4G17r7O3T8EFhM7dnnl7svcfW+w+Cyx6zaKKcxxmAjc5zHPAoeb2VH5Dszd3/JggkJ33w68TOzK9lJRlOOW4EzgdXcv1OX3bbj7U8A7CavD5Kms/kdLKuG3I9XUDon6eOz6AILfR+Y5rtOAze7+WortDiwzsxXB1BKFMjX4Gj0/xdfFsMczn64m1gJMplDHLcxxKPqxMrMBwMnAc0k2n2pmK83sYTM7sYBhpXuPin7ciF0XlKoxVqzjBuHyVFbHL3I3QDGzx4CPJdk0w91/napaknV5HW8aMs7Lab91P8bdN5nZkcCjZrYm+MTPW2zAj4GbiB2fm4h1OV2duIskdXNyPMMcNzObAewFFqbYTV6OW7Jwk6wLO61IQZjZocAvga+7+/sJm/9MrLtiR3Cu5iFgUIFCS/ceFfu4dQUuAL6VZHMxj1tYWR2/yCV8dz8ri2phpn8A2GxmR7n7W8HXxy3ZxAjp4zSzzsBFkHoCJnffFPzeYmYPEvua1uHEFfYYmtk9xM33FCfs8cxYiOP2L8D5wJkedFYm2UdejlsSHZlWJO/MrAuxZL/Q3dvcFiv+A8Ddl5rZ3WbWy93zPiFdiPeoaMctMAH4s7tvTtxQzOMWCJOnsjp+5dKlswS4zMz+ycwGEvs0fj5FuX8JHv8LkOobQy6cBaxx9+ZkG83sEDPr2fKY2AnLF5OVzaWEftILUzxnUabEsNjNcqYBF7h70psVFPi4dWRakbwKzg/9BHjZ3W9PUeZjQTnMbCSx//dtBYgtzHtUlOMWJ+W372Idtzhh8lR2/6OFOBOdqx9iCaoZ+AewGXgkbtsMYmetXwEmxK2fRzCiB6gGHgdeC34fkcdYFwBTEtYdDSwNHn+c2Jn1lcBqYl0ahTiG/w2sAl4I/kCOSowtWP4csZEfrxcwtrXE+iWbgp85xT5uyY4DMKXlvSX21Xp2sH0VcaPH8hzXZ4h9hX8h7nh9LiG2qcExWknsJPinCxRb0vcoCscteO4exBL4YXHrinLciH3ovAXsCXLbNanyVC7+RzW1gohIhSiXLh0REUlDCV9EpEIo4YuIVAglfBGRCqGELyJSIZTwRUQqhBK+iEiF+P9y6ZcU/uE+vwAAAABJRU5ErkJggg==)

然后我们看看样本集生成的二维正态分布，代码如下：

```python
fig = plt.figure()   # 生成图像句柄
ax = Axes3D(fig, rect=[0,0,1,1], elev=30, azim=20)   # 生成坐标句柄
ax.scatter(x1_list, x2_list,z_list, marker='o', c='#00CED1')
plt.show()
```

![img](data:image/png;base64,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)

