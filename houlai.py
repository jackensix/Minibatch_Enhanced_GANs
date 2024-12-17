import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# from keras.optimizers import Adam
# from keras.models import Sequential, load_model
# from keras.layers import Dense, Dropout, LeakyReLU, Reshape
# from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import get_file
import matplotlib.pyplot as plt
# import seaborn as sns
from zipfile import ZipFile
# from keras.utils.vis_utils import plot_model
import os
# %matplotlib inline
## Glimmer007 
# 
from keras.callbacks import EarlyStopping,CSVLogger
import tensorflow as tf

# # 固定随机种子
# np.random.seed(42)
# tf.random.set_seed(42)


from keras import backend as K

# 自定义判别器损失函数，添加L1和L2正则化
def custom_loss_with_l1_l2(y_true, y_pred):
    y_true = K.cast(y_true, dtype='float32')
    binary_crossentropy_loss = K.binary_crossentropy(y_true, y_pred)
    l1_lambda = 0.001  # L1正则化强度
    l2_lambda = 0.001  # L2正则化强度
    l1_l2_loss = l1_lambda * sum([K.sum(K.abs(w)) for w in discriminator.trainable_weights])
    l1_l2_loss += l2_lambda * sum([K.sum(K.square(w)) for w in discriminator.trainable_weights])
    total_loss = binary_crossentropy_loss + l1_l2_loss
    return total_loss

# 自定义生成器损失函数，添加L1和L2正则化
def generator_loss_with_l1_l2(y_true, y_pred):
    y_true = K.cast(y_true, dtype='float32')
    binary_crossentropy_loss = K.binary_crossentropy(y_true, y_pred)
    l1_lambda = 0.001  # L1正则化强度
    l2_lambda = 0.001  # L2正则化强度
    l1_l2_loss = l1_lambda * sum([K.sum(K.abs(w)) for w in generator.trainable_weights])
    l1_l2_loss += l2_lambda * sum([K.sum(K.square(w)) for w in generator.trainable_weights])
    total_loss = binary_crossentropy_loss + l1_l2_loss
    return total_loss




path = "DataSet-master.zip"
filespath = os.path.dirname(path)+"/NSL-KDD/"
with ZipFile(path, 'r') as zipObj:
    zipObj.extractall(filespath)
    print(zipObj.namelist())

filespath = "DataSet-master\\NSL-KDD\\"
df_train = pd.read_csv(filepath_or_buffer=filespath+'KDDTrain+.txt',delimiter=',', header=None)
df_test = pd.read_csv(filepath_or_buffer=filespath+'KDDTest+.txt',delimiter=',', header=None)


# The CSV file has no column heads, so add them
df_train.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome',
    'difficulty'
]
df_test.columns = df_train.columns


print(f"df_train.shape = {df_train.shape}")
print(f"df_test.shape = {df_test.shape}")


df_obj_col = df_train.select_dtypes(include='object').columns
print(df_obj_col)

plt.figure(figsize=(18, 5))
plt.subplot(131)
df_train["protocol_type"].value_counts().plot(kind='bar', label='protocol type')
plt.legend()
plt.subplot(132)
df_train['service'].value_counts().head(10).plot(kind='bar')
plt.legend()
plt.subplot(133)
df_train["flag"].value_counts().plot(kind='bar')
plt.legend()
# plt.show()
plt.savefig("1.jpg")

df_train['label'] = np.where(df_train['outcome'].str.contains('normal'), 0, 1)
df_test['label'] = np.where(df_test['outcome'].str.contains('normal'), 0, 1)

df_train_obj = df_train.iloc[:, :-3].select_dtypes(include='object')
df_train_num = df_train.iloc[:, :-3].select_dtypes(exclude='object')

print(f"shape of numeric features: {df_train_num.shape}")
print(f"shape of object features: {df_train_obj.shape}")

df_test_obj = df_test.iloc[:, :-3].select_dtypes(include='object')
df_test_num = df_test.iloc[:, :-3].select_dtypes(exclude='object')

print(f"shape of numeric features: {df_test_num.shape}")
print(f"shape of object features: {df_test_obj.shape}")

enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(df_train_obj)
# X_train = np.c_[df_train_num, X_train_enc.toarray()]
X_train = np.c_[df_train_num, X_train_enc.toarray()][df_train.outcome == 'normal']
X_test_enc = enc.transform(df_test_obj)
X_test = np.c_[df_test_num, X_test_enc.toarray()]
# X_test_enc = enc.transform(df_test_obj)
X_test_normal = np.c_[df_test_num, X_test_enc.toarray()][df_test.outcome == 'normal']
X_test_abnormal = np.c_[df_test_num, X_test_enc.toarray()][df_test.outcome != 'normal']


# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


print(f"shape of X_train: {X_train.shape}")
print(f"shape of X_test: {X_test.shape}")
print(f"shape of X_test_normal: {X_test_normal.shape}")
print(f"shape of X_test_abnormal: {X_test_abnormal.shape}")


outlier_fence_95 = np.percentile(X_train, 95, axis=0)
X_train_new = X_train.copy()
for index, fence in enumerate(outlier_fence_95):
    boolarr = X_train_new[:,index] <= fence
    X_train_new = X_train_new[boolarr]
# X_train_new.shape


# from sklearn.preprocessing import RobustScaler, MinMaxScaler
# # scaler = RobustScaler()
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_new)
X_test_scaled = scaler.transform(X_test)
X_test_normal_scaled = scaler.transform(X_test_normal)
X_test_abnormal_scaled = scaler.transform(X_test_abnormal)
y_train = np.zeros((X_train_scaled.shape[0],1))
y_test = df_test['label']





from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, concatenate
from keras.layers import LeakyReLU
# from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
# from keras.optimizers import Adam
from keras.optimizers import Adam
from keras import losses
# from keras.utils import to_categorical
from keras.initializers import RandomNormal
from keras.utils.vis_utils import plot_model
import keras.backend as K

import matplotlib.pyplot as plt

optimizer = Adam(0.0001, 0.5)
instance_shape = 122
latent_dim = 10
# initializer = RandomNormal(mean=0., stddev=1.)
initializer = 'he_normal'
encoder = Sequential()
encoder.add(Dense(32, activation='relu', input_shape=(instance_shape,), kernel_initializer=initializer))
encoder.add(Dense(latent_dim, activation='relu'))
encoder.summary()

plot_model(encoder, to_file='e_model_plot.png', show_shapes=True, show_layer_names=True)              


latent_dim = 10
generator = Sequential()
generator.add(Dense(32, activation='relu', input_dim=latent_dim, kernel_initializer=initializer))
generator.add(Dense(instance_shape, activation='sigmoid'))
generator.summary()
plot_model(generator, to_file='g_model_plot.png', show_shapes=True, show_layer_names=True)













from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU, concatenate, Layer
from keras.optimizers import Adam
from keras.utils import plot_model
import tensorflow as tf
from keras.regularizers import l2
# 定义潜在向量维度和输入形状
latent_dim = 10  # 潜在向量的维度
instance_shape = (122,)  # 实例输入的形状，作为元组

# 初始化方法和优化器
# initializer = 'he_normal'
# optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

# 自定义 Minibatch Discrimination 层
class MinibatchDiscrimination(Layer):
    def __init__(self, num_kernels=50, dim_per_kernel=20, **kwargs):
        super(MinibatchDiscrimination, self).__init__(**kwargs)
        self.num_kernels = num_kernels
        self.dim_per_kernel = dim_per_kernel
        self.initializer = 'he_normal'

    def build(self, input_shape):
        self.M = self.add_weight(
            shape=(input_shape[-1], self.num_kernels * self.dim_per_kernel),
            initializer=self.initializer,
            trainable=True,
            name='M'
        )

    def call(self, x):
        # Matrix multiplication followed by reshaping
        M = tf.matmul(x, self.M)
        M = tf.reshape(M, (-1, self.num_kernels, self.dim_per_kernel))

        # Normalize M for cosine similarity
        M_norm = tf.nn.l2_normalize(M, axis=2)

        # Compute cosine similarity between all samples
        cosine_similarity = tf.matmul(M_norm, M_norm, transpose_b=True)

        # Create a mask to exclude self-comparison
        batch_size = tf.shape(x)[0]
        mask = 1.0 - tf.eye(self.num_kernels)  # Only for num_kernels, not batch size
        mask = tf.reshape(mask, (1, self.num_kernels, self.num_kernels))  # Add batch dimension
        mask = tf.tile(mask, [batch_size, 1, 1])  # Tile to match batch size
        
        # Apply the mask to cosine similarity
        masked_similarity = cosine_similarity * mask

        # Aggregate minibatch features
        minibatch_features = tf.reduce_sum(masked_similarity, axis=2)
        return minibatch_features

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_kernels)






# 定义输入
z = Input(shape=(latent_dim,))
img = Input(shape=instance_shape)  # 直接使用元组格式

# 合并输入
d_in = concatenate([z, img])

# 添加全连接层和激活函数
model = Dense(32, kernel_initializer=initializer)(d_in)
model = LeakyReLU(alpha=0.2)(model)
model = Dropout(0.2)(model)
# 添加自定义的 Minibatch Discrimination 层
minibatch_features = MinibatchDiscrimination(num_kernels=5, dim_per_kernel=3)(model)

# 合并原始特征和 Minibatch Discrimination 特征
combined = concatenate([model, minibatch_features])

# 输出层
validity = Dense(1, activation="sigmoid")(combined)

# 构建判别器模型
discriminator = Model([z, img], validity)

# 设置判别器为可训练
discriminator.trainable = True

# 编译模型
discriminator.compile(loss=custom_loss_with_l1_l2,
                      optimizer=optimizer,
                      metrics=['accuracy'])

# 打印模型总结
discriminator.summary()

# 可视化模型结构
plot_model(discriminator, to_file='d_model_plot.png', show_shapes=True, show_layer_names=True)











discriminator.trainable = False
# Generate traffic record from sampled noise
z = Input(shape=(latent_dim,))
img_ = generator(z)
# Encode traffic records to generate latent space
# img = Input(shape=(instance_shape,))
img = Input(shape=instance_shape)
z_ = encoder(img)
# Encode traffic records to generate latent space
img = Input(shape=instance_shape) 
z_ = encoder(img)

# Latent -> img is fake, and img -> latent is valid
fake = discriminator([z, img_])
valid = discriminator([z_, img])
bigan_generator = Model([z, img], [fake, valid])
# 让生成器也使用自定义损失函数
# 编译生成器模型
bigan_generator.compile(loss=[generator_loss_with_l1_l2, generator_loss_with_l1_l2],  # 使用L1和L2正则化的自定义损失
                        optimizer=optimizer)

bigan_generator.summary()
plot_model(bigan_generator, to_file='biganG_model_plot.png', show_shapes=True, show_layer_names=True)


# 早期停止回调
# early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# X_train = X_train_scaled
# Adversarial ground truths
batch_size=64
valid = np.zeros((batch_size, 1))
fake = np.ones((batch_size, 1))
epoches = 1000
sample_interval=1

# 初始化列表来保存损失值
d_losses = []  # 判别器损失
g_losses = []  # 生成器损失
for epoch in range(epoches):
    # ---------------------
    #  Train Discriminator
    # ---------------------
    discriminator.trainable = True
    # Sample noise and generate img
    z = np.random.normal(size=(batch_size, latent_dim))
    imgs_ = generator.predict(z)

    # Select a random batch of images and encode
    idx = np.random.randint(0, X_train_scaled.shape[0], batch_size)
    imgs = X_train_scaled[idx]
    z_ = encoder.predict(imgs)

    # Train the discriminator (img -> z is valid, z -> img is fake)
    d_loss_real = discriminator.train_on_batch([z_, imgs], valid)
    d_loss_fake = discriminator.train_on_batch([z, imgs_], fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    # ---------------------
    #  Train Generator
    # ---------------------

    # 保存判别器的损失
    d_losses.append(d_loss[0])

    # Train the generator (z -> img is valid and img -> z is is invalid)
    discriminator.trainable = False
    import pdb
    # pdb.set_trace()
    for i in range(5):   
        g_loss = bigan_generator.train_on_batch([z, imgs], [valid, fake])

    # 保存生成器的损失
    g_losses.append(g_loss[0])

    # Plot the progress
    if epoch % sample_interval == 0:
        print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))
    # Plot the progress
    # if epoch % sample_interval == 0:
    #     print ("%d [D loss: %f, acc: %.2f%%]" % (epoch, d_loss[0], 100*d_loss[1]))

# 训练结束后保存损失值到文件（加入 Minibatch 的代码）
np.save('d_losses_with_minibatch.npy', d_losses)
np.save('g_losses_with_minibatch.npy', g_losses)


# evaluate the discriminator
z_test_normal = encoder.predict(X_test_normal_scaled)
y_test_normal = np.zeros((X_test_normal_scaled.shape[0], 1))
discriminator.evaluate([z_test_normal, X_test_normal_scaled], y_test_normal)


z_test_abnormal = encoder.predict(X_test_abnormal_scaled)
y_test_abnormal = np.ones((X_test_abnormal_scaled.shape[0], 1))
discriminator.evaluate([z_test_abnormal, X_test_abnormal_scaled], y_test_abnormal)


z_test = encoder.predict(X_test_scaled)
discriminator.evaluate([z_test, X_test_scaled], y_test)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred = discriminator.predict([z_test, X_test_scaled])
cm = confusion_matrix(y_test, y_pred>0.5)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred>0.5).ravel()
ConfusionMatrixDisplay(confusion_matrix=cm).plot()


Precision = tp / (tp + fp)
Recall = tp / (tp + fn)
Accuracy = (tp + tn) / (tp + tn + fp + fn)
F1 = 2 * Precision * Recall / (Precision + Recall)
print(f"Precision = {Precision}")
print(f"Recall = {Recall}")
print(f"Accuracy = {Accuracy}")
print(f"F1 = {F1}")


from sklearn.metrics import roc_auc_score, roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
np.save('fpr_with_minibatch.npy', fpr)
np.save('tpr_with_minibatch.npy', tpr)
print()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:0.3f}')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
# plt.show()
plt.savefig("mix.jpg")

import json

# 同理在第二个代码中保存性能指标
metrics_minibatch = {
    'Precision': Precision,
    'Recall': Recall,
    'Accuracy': Accuracy,
    'F1': F1,
    'ROC_AUC': roc_auc
}

with open('results_with_minibatch.json', 'w') as f:
    json.dump(metrics_minibatch, f)