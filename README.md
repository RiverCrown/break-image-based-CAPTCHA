## 验证码识别

### 简介

这是比赛的赛题，总共给出了五种验证码类型。

由于时间紧凑，源码中基本上没有注释，博客中也没有对代码进行讲解；但代码核心思路和基本框架都是参照 [斗大的熊猫](http://blog.topspeedsnail.com/archives/10858) 的博客，如果看懂了 **斗大的熊猫** 的文章，前四题基本上采用一样的思路就可以拟合，如果想达到更好的效果就采用 **更好的网络结构** 并不断地 **调参** 。

第五题思路就不能像前面四题那样直接通过神经网络学习到 **多分类** ，只能通过先学习到 **嵌套（embedding）**，之后再根据 embedding 之间的距离来识别两张图片是否相似（相似度学习）。

验证码类型、解题思路和模型结构主要放在我的 [博客](https://rivercrown.github.io/2018/06/25/验证码识别-image-based/#more) 中。

### 目录结构

dataset/ 目录中对于每个类型的验证码都给出了 10 张样例。

上传的项目中有五个关于代码的文件夹，每个文件夹对应一道验证码识别题型，由于 **模型过大** 所以没有传上来。目录结构如下所示：

.  
├── README.md  
├── captcha1  
│   ├── mappings.txt  
│   ├── my_cnn.py  
│   ├── my_get_captcha.py  
│   ├── params.json  
│   ├── pre_solve.py  
│   ├── pre_solved  
│   └── split_record.txt  
├── captcha2  
│   ├── my_cnn.py  
│   ├── my_get_captcha.py  
│   └── params.json  
├── captcha3  
│   ├── my_cnn.py  
│   ├── my_get_captcha.py  
│   └── params.json  
├── captcha4  
│   ├── my_cnn.py  
│   ├── my_get_captcha.py  
│   └── params.json  
└── captcha5  
    ├── calculate_distance.py  
    ├── my_cnn.py  
    ├── my_get_batch.py  
    ├── params.json  
    ├── pre_solve.py  
    ├── pre_solved  
    └── triplet_loss.py  

每个文件夹中都 **至少** 有 params.json、my_cnn.py 和 my_get_captcha.py 这三个文件。他们的作用分别是：

+ params.json：用来配置数据集所在目录、预处理图片存储目录和模型目录等等。
+ my_cnn.py：该文件中包含网络结构代码、训练代码和识别代码。
+ my_get_captcha.py：该文件包括获取数据集的代码。

在代码执行过程中产生或者需要自己创建的文件/文件夹有：

+ pre_solved/：该文件夹包含预处理过后的数据。
+ model/：该文件夹包含训练得到的模型。
+ mappings.txt：该文件包含最后的识别结果。

第一题的结构：

+ pre_solve.py：包含预处理代码，将预处理后的数据存入 ./pre_solved 目录中（自己创建）。
+ split_record.txt：第一题是不定长验证码，图片分割长度结果存入该文件中，便于训练和识别的时候读取。

第五题的结构：

+ calculate_distance.py：包含计算嵌套（embedding）之间距离的代码。
+ pre_solve.py：包含预处理代码，将预处理后的数据存入 ./pre_solved 目录中（自己创建）。
+ triplet_loss.py：包含计算三元组损失的代码。（[代码来源](https://omoindrot.github.io/triplet-loss)）
