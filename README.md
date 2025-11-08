<img width="815" height="563" alt="image" src="https://github.com/user-attachments/assets/947e7591-f499-4148-a046-533948b4c0df" />
文件分为三个部分
train：代表训练文件夹
models：代表本文使用的对比模型和本文搭建的模型
data：代表所用的数据库和预处理代码
other：一些文章中出现的图的代码
使用流程：下载data 使用preprocess去得到预处理后的数据，然后使用train去训练数据，从models导入你想要的模型
最简单的使用方法，从models导入MCTMNet，然后model = MCTMNet(num_channels=61, num_timesteps=num_timesteps, num_classes=4)
如果你对文章或者模型感兴趣有任何的问题，可以发邮件给我：327256279@qq.com

The document is divided into three parts
"train: Represents the training folder.
models: Represents the contrast models used in this article and the models built in this article
data: Represents the database used and the preprocessing code
other: Codes for some figures that appear in articles

Usage process: Download data and use preprocess to obtain the preprocessed data. Then, use train to train the data and import the model you want from models
The simplest usage method is to import MCTMNet from models, and then model = MCTMNet(num_channels=61, num_timesteps=num_timesteps, num_classes=4)
If you have any questions about the article or the model, you can email me at 327256279@qq.com
