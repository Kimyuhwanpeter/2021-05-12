# -*- coding:utf-8 -*-
import tensorflow as tf

# self attention 유사도? 생각해보면서 연구하고 코딩하기

l2 = tf.keras.regularizers.l2

def fix_GL_network(input_shape=(128, 88, 1), weight_decay=0.00001, num_classes=86):

    h = inputs = tf.keras.Input(input_shape)

    crop_1 = tf.image.crop_to_bounding_box(h, 0, 0, 22, 88)
    crop_2 = tf.image.crop_to_bounding_box(h, 22, 0, 48, 88)
    crop_3 = tf.image.crop_to_bounding_box(h, 70, 0, 58, 88)

    ###########################################################################################
    crop_1 = tf.keras.layers.ZeroPadding2D((2, 3))(crop_1)
    crop_1 = tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=(5,7),
                                    strides=1,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_1)
    crop_1 = tf.keras.layers.BatchNormalization()(crop_1)
    crop_1 = tf.keras.layers.LeakyReLU()(crop_1)

    crop_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(crop_1)

    crop_1 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(5,7),
                                    strides=2,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_1)
    crop_1 = tf.keras.layers.BatchNormalization()(crop_1)
    crop_1 = tf.keras.layers.LeakyReLU()(crop_1)

    crop_1_max = tf.keras.layers.GlobalMaxPool2D()(crop_1)
    crop_1_max = tf.keras.layers.Dense(64/16)(crop_1_max)
    crop_1_max = tf.keras.layers.ReLU()(crop_1_max)
    crop_1_max = tf.keras.layers.Dense(64)(crop_1_max)

    crop_1_avg = tf.keras.layers.GlobalAveragePooling2D()(crop_1)
    crop_1_avg = tf.keras.layers.Dense(64/16)(crop_1_avg)
    crop_1_avg = tf.keras.layers.ReLU()(crop_1_avg)
    crop_1_avg = tf.keras.layers.Dense(64)(crop_1_avg)

    crop_1_sum = crop_1_max + crop_1_avg
    crop_1_sum = tf.nn.sigmoid(crop_1_sum)
    crop_1_sum = tf.expand_dims(crop_1_sum, 1)
    crop_1_sum = tf.expand_dims(crop_1_sum, 1)
    crop_1_ = tf.math.multiply(crop_1, crop_1_sum)
    
    crop_1_multi = tf.concat([ tf.reduce_max(crop_1_, -1, keepdims=True), tf.reduce_mean(crop_1_, -1, keepdims=True) ], -1)
    crop_1_multi = tf.keras.layers.Conv2D(filters=1,
                                    kernel_size=(5,7),
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_1_multi)
    crop_1_multi = tf.keras.layers.BatchNormalization()(crop_1_multi)
    crop_1 = tf.multiply(crop_1_multi, crop_1_)
    crop_1 = tf.nn.sigmoid(crop_1)       
    ###########################################################################################

    ###########################################################################################
    crop_2 = tf.keras.layers.ZeroPadding2D((2, 3))(crop_2)
    crop_2 = tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=(5,7),
                                    strides=1,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_2)
    crop_2 = tf.keras.layers.BatchNormalization()(crop_2)
    crop_2 = tf.keras.layers.LeakyReLU()(crop_2)

    crop_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(crop_2)

    crop_2 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(5,7),
                                    strides=2,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_2)
    crop_2 = tf.keras.layers.BatchNormalization()(crop_2)
    crop_2 = tf.keras.layers.LeakyReLU()(crop_2)

    crop_2_max = tf.keras.layers.GlobalMaxPool2D()(crop_2)
    crop_2_max = tf.keras.layers.Dense(64/16)(crop_2_max)
    crop_2_max = tf.keras.layers.ReLU()(crop_2_max)
    crop_2_max = tf.keras.layers.Dense(64)(crop_2_max)

    crop_2_avg = tf.keras.layers.GlobalAveragePooling2D()(crop_2)
    crop_2_avg = tf.keras.layers.Dense(64/16)(crop_2_avg)
    crop_2_avg = tf.keras.layers.ReLU()(crop_2_avg)
    crop_2_avg = tf.keras.layers.Dense(64)(crop_2_avg)

    crop_2_sum = crop_2_max + crop_2_avg
    crop_2_sum = tf.nn.sigmoid(crop_2_sum)
    crop_2_sum = tf.expand_dims(crop_2_sum, 1)
    crop_2_sum = tf.expand_dims(crop_2_sum, 1)
    crop_2_ = tf.math.multiply(crop_2, crop_2_sum)

    crop_2_multi = tf.concat([ tf.reduce_max(crop_2_, -1, keepdims=True), tf.reduce_mean(crop_2_, -1, keepdims=True) ], -1)
    crop_2_multi = tf.keras.layers.Conv2D(filters=1,
                                    kernel_size=(5,7),
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_2_multi)
    crop_2_multi = tf.keras.layers.BatchNormalization()(crop_2_multi)
    crop_2 = tf.multiply(crop_2_multi, crop_2_)
    crop_2 = tf.nn.sigmoid(crop_2)
    ###########################################################################################

    ###########################################################################################
    crop_3 = tf.keras.layers.ZeroPadding2D((2, 3))(crop_3)
    crop_3 = tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=(5,7),
                                    strides=1,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_3)
    crop_3 = tf.keras.layers.BatchNormalization()(crop_3)
    crop_3 = tf.keras.layers.LeakyReLU()(crop_3)

    crop_3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(crop_3)

    crop_3 = tf.keras.layers.ZeroPadding2D((1, 3))(crop_3)
    crop_3 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(5,7),
                                    strides=2,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_3)
    crop_3 = tf.keras.layers.BatchNormalization()(crop_3)
    crop_3 = tf.keras.layers.LeakyReLU()(crop_3)

    crop_3_max = tf.keras.layers.GlobalMaxPool2D()(crop_3)
    crop_3_max = tf.keras.layers.Dense(64/16)(crop_3_max)
    crop_3_max = tf.keras.layers.ReLU()(crop_3_max)
    crop_3_max = tf.keras.layers.Dense(64)(crop_3_max)

    crop_3_avg = tf.keras.layers.GlobalAveragePooling2D()(crop_3)
    crop_3_avg = tf.keras.layers.Dense(64/16)(crop_3_avg)
    crop_3_avg = tf.keras.layers.ReLU()(crop_3_avg)
    crop_3_avg = tf.keras.layers.Dense(64)(crop_3_avg)

    crop_3_sum = crop_3_max + crop_3_avg
    crop_3_sum = tf.nn.sigmoid(crop_3_sum)
    crop_3_sum = tf.expand_dims(crop_3_sum, 1)
    crop_3_sum = tf.expand_dims(crop_3_sum, 1)
    crop_3_ = tf.math.multiply(crop_3, crop_3_sum)

    crop_3_multi = tf.concat([ tf.reduce_max(crop_3_, -1, keepdims=True), tf.reduce_mean(crop_3_, -1, keepdims=True) ], -1)
    crop_3_multi = tf.keras.layers.Conv2D(filters=1,
                                    kernel_size=(5,7),
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_3_multi)
    crop_3_multi = tf.keras.layers.BatchNormalization()(crop_3_multi)
    crop_3 = tf.multiply(crop_3_multi, crop_3_)
    crop_3 = tf.nn.sigmoid(crop_3)
    ###########################################################################################
    
    crop = tf.concat([crop_1, crop_2, crop_3], 1)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)

    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h_max = tf.keras.layers.GlobalMaxPool2D()(h)
    h_max = tf.keras.layers.Dense(64/16)(h_max)
    h_max = tf.keras.layers.ReLU()(h_max)
    h_max = tf.keras.layers.Dense(64)(h_max)

    h_avg = tf.keras.layers.GlobalAveragePooling2D()(h)
    h_avg = tf.keras.layers.Dense(64/16)(h_avg)
    h_avg = tf.keras.layers.ReLU()(h_avg)
    h_avg = tf.keras.layers.Dense(64)(h_avg)

    h_sum = h_max + h_avg
    h_sum = tf.nn.sigmoid(h_sum)
    h_sum = tf.expand_dims(h_sum, 1)
    h_sum = tf.expand_dims(h_sum, 1)
    h_ = tf.math.multiply(h, h_sum)

    h_multi = tf.concat([ tf.reduce_max(h_, -1, keepdims=True), tf.reduce_mean(h_, -1, keepdims=True) ], -1)
    h_multi = tf.keras.layers.Conv2D(filters=1,
                                kernel_size=(5,7),
                                padding="same",
                                use_bias=False,
                                kernel_regularizer=l2(weight_decay))(h_multi)
    h_multi = tf.keras.layers.BatchNormalization()(h_multi)
    h = tf.multiply(h_multi, h_)
    h = tf.nn.sigmoid(h)

    h = tf.concat([h, crop], -1)

    h = tf.keras.layers.ZeroPadding2D((2,2))(h)
    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=5,
                               strides=2,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)

    h = tf.keras.layers.ZeroPadding2D((2,2))(h)
    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=5,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    ##################################################################################################  dual attention (position)
    h_position_att_B = tf.keras.layers.Conv2D(filters=128 // 8,
                                              kernel_size=1,
                                              strides=1,
                                              use_bias=False,
                                              kernel_regularizer=l2(weight_decay))(h)   # query 영향을 받는 feature
    h_position_att_B =  tf.keras.layers.Reshape((8*6, 128 // 8))(h_position_att_B)

    h_position_att_C = tf.keras.layers.Conv2D(filters=128 // 8,
                                              kernel_size=1,
                                              strides=1,
                                              use_bias=False,
                                              kernel_regularizer=l2(weight_decay))(h)   # key 영향을 주는 feature
    h_position_att_C = tf.keras.layers.Reshape((128 // 8, 8*6))(h_position_att_C)

    e = tf.matmul(h_position_att_B, h_position_att_C)   # 유사도 계산    (query와 key의 내적은 유사도를 측정하는것과 같다)
    attention = tf.nn.softmax(e, -1)
    attention = tf.reshape(attention, [-1, attention.shape[2], attention.shape[1]])


    h_position_att_D = tf.keras.layers.Conv2D(filters=128,
                                              kernel_size=1,
                                              strides=1,
                                              use_bias=False,
                                              kernel_regularizer=l2(weight_decay))(h)   # value 이 영향들에 대한 가중치
    h_position_att_D = tf.keras.layers.Reshape((128, 8*6))(h_position_att_D)
    h_position_out = tf.matmul(h_position_att_D, attention)
    h_position_out = tf.keras.layers.Reshape((8, 6, 128))(h_position_out)
    h_position_out = h_position_out + h
    ##################################################################################################

    ##################################################################################################
    h_channel_att_A = tf.keras.layers.Reshape((128, 8*6))(h)    # query
    h_channel_att_B = tf.keras.layers.Reshape((8*6, 128))(h)    # key
    e2 = tf.matmul(h_channel_att_A, h_channel_att_B)    # query 와 key의 유사도 계산
    e2 = tf.reduce_max(e2, -1, keepdims=True) - e2
    attention2 = tf.nn.softmax(e2, -1)

    value = tf.keras.layers.Reshape((128, 8*6))(h)

    h_channel_out = tf.matmul(attention2, value)
    h_channel_out = tf.keras.layers.Reshape((8, 6, 128))(h_channel_out)
    h_channel_out = h_channel_out + h
    ##################################################################################################

    h = h_channel_out + h_position_out

    h = tf.keras.layers.GlobalMaxPool2D()(h)

    h = tf.keras.layers.Dense(1024)(h)
    h = tf.keras.layers.LeakyReLU()(h)
    h = tf.keras.layers.Dropout(0.5)(h)

    h = tf.keras.layers.Dense(1024)(h)
    h = tf.keras.layers.LeakyReLU()(h)
    h = tf.keras.layers.Dropout(0.5)(h)

    #h1 = tf.keras.layers.Dense(num_classes)(h)  # 이 출력을 조금 바꿔보자

    h1 = tf.keras.layers.Dense(10)(h)   # 10 feature에 대한 key
    h1 = tf.expand_dims(h1, 1)  # [B, 1, 10]

    h2 = tf.keras.layers.Dense(9)(h)    # 9 feature에 대한 key
    h2 = tf.expand_dims(h2, 1)  # [B,1,9]

    h3 = tf.keras.layers.Dense(90, use_bias=False)(h)
    h3 = tf.keras.layers.Reshape((9, 10))(h3)   # value?

    h_split_1 = tf.reduce_max(h3, 2, keepdims=True)  # [B, 9, 1] , 9 feature에 대한 query
    h_split_2 = tf.reduce_max(h3, 1, keepdims=True)    # [B, 1, 10] , 10 feature에 대한 query
    h_split_2 = tf.reshape(h_split_2, [-1, 10, 1])  # [B, 10, 1]
    
    h_position_1 = tf.matmul(h_split_1, h2) # [B, 9, 9]
    h_position_1 = tf.nn.softmax(h_position_1, -1)
    h_position_1 = tf.matmul(h_position_1, h3)  # [B, 9, 10]

    e3 = tf.matmul(h_split_2, h1) # [B, 10, 10]
    e3 = tf.reduce_max(e3, -1, keepdims=True) - e3 # [B, 10, 10]
    h_attention = tf.nn.softmax(e3, -1)    # [B, 10, 10]
    h_channel_2 = tf.matmul(h3, h_attention)  # [B, 9, 10]

    h = (h_position_1 + h_channel_2)   # self attention을 나름대로 추가해준것! train 파일도 같이 고쳐야하ㅣㄴ다!! 월요일에 고쳐!! 기억해!!!!!!!!!!!!!!!!!!!!!!!!

    return tf.keras.Model(inputs=inputs, outputs=h)

def fix_GL_network_2(input_shape=(128, 88, 1), weight_decay=0.00001, num_classes=86):

    h = inputs = tf.keras.Input(input_shape)

    crop_1 = tf.image.crop_to_bounding_box(h, 0, 0, 22, 88)
    crop_2 = tf.image.crop_to_bounding_box(h, 22, 0, 48, 88)
    crop_3 = tf.image.crop_to_bounding_box(h, 70, 0, 58, 88)

    ###########################################################################################
    crop_1 = tf.keras.layers.ZeroPadding2D((2, 3))(crop_1)
    crop_1 = tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=(5,7),
                                    strides=1,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_1)
    crop_1 = tf.keras.layers.BatchNormalization()(crop_1)
    crop_1 = tf.keras.layers.LeakyReLU()(crop_1)

    crop_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(crop_1)

    crop_1 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(5,7),
                                    strides=2,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_1)
    crop_1 = tf.keras.layers.BatchNormalization()(crop_1)
    crop_1 = tf.keras.layers.LeakyReLU()(crop_1)

    crop_1_max = tf.keras.layers.GlobalMaxPool2D()(crop_1)
    crop_1_max = tf.keras.layers.Dense(64/16)(crop_1_max)
    crop_1_max = tf.keras.layers.ReLU()(crop_1_max)
    crop_1_max = tf.keras.layers.Dense(64)(crop_1_max)

    crop_1_avg = tf.keras.layers.GlobalAveragePooling2D()(crop_1)
    crop_1_avg = tf.keras.layers.Dense(64/16)(crop_1_avg)
    crop_1_avg = tf.keras.layers.ReLU()(crop_1_avg)
    crop_1_avg = tf.keras.layers.Dense(64)(crop_1_avg)

    crop_1_sum = crop_1_max + crop_1_avg
    crop_1_sum = tf.nn.sigmoid(crop_1_sum)
    crop_1_sum = tf.expand_dims(crop_1_sum, 1)
    crop_1_sum = tf.expand_dims(crop_1_sum, 1)
    crop_1_ = tf.math.multiply(crop_1, crop_1_sum)
    
    crop_1_multi = tf.concat([ tf.reduce_max(crop_1_, -1, keepdims=True), tf.reduce_mean(crop_1_, -1, keepdims=True) ], -1)
    crop_1_multi = tf.keras.layers.Conv2D(filters=1,
                                    kernel_size=(5,7),
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_1_multi)
    crop_1_multi = tf.keras.layers.BatchNormalization()(crop_1_multi)
    crop_1 = tf.multiply(crop_1_multi, crop_1_)
    crop_1 = tf.nn.sigmoid(crop_1)       
    ###########################################################################################

    ###########################################################################################
    crop_2 = tf.keras.layers.ZeroPadding2D((2, 3))(crop_2)
    crop_2 = tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=(5,7),
                                    strides=1,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_2)
    crop_2 = tf.keras.layers.BatchNormalization()(crop_2)
    crop_2 = tf.keras.layers.LeakyReLU()(crop_2)

    crop_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(crop_2)

    crop_2 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(5,7),
                                    strides=2,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_2)
    crop_2 = tf.keras.layers.BatchNormalization()(crop_2)
    crop_2 = tf.keras.layers.LeakyReLU()(crop_2)

    crop_2_max = tf.keras.layers.GlobalMaxPool2D()(crop_2)
    crop_2_max = tf.keras.layers.Dense(64/16)(crop_2_max)
    crop_2_max = tf.keras.layers.ReLU()(crop_2_max)
    crop_2_max = tf.keras.layers.Dense(64)(crop_2_max)

    crop_2_avg = tf.keras.layers.GlobalAveragePooling2D()(crop_2)
    crop_2_avg = tf.keras.layers.Dense(64/16)(crop_2_avg)
    crop_2_avg = tf.keras.layers.ReLU()(crop_2_avg)
    crop_2_avg = tf.keras.layers.Dense(64)(crop_2_avg)

    crop_2_sum = crop_2_max + crop_2_avg
    crop_2_sum = tf.nn.sigmoid(crop_2_sum)
    crop_2_sum = tf.expand_dims(crop_2_sum, 1)
    crop_2_sum = tf.expand_dims(crop_2_sum, 1)
    crop_2_ = tf.math.multiply(crop_2, crop_2_sum)

    crop_2_multi = tf.concat([ tf.reduce_max(crop_2_, -1, keepdims=True), tf.reduce_mean(crop_2_, -1, keepdims=True) ], -1)
    crop_2_multi = tf.keras.layers.Conv2D(filters=1,
                                    kernel_size=(5,7),
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_2_multi)
    crop_2_multi = tf.keras.layers.BatchNormalization()(crop_2_multi)
    crop_2 = tf.multiply(crop_2_multi, crop_2_)
    crop_2 = tf.nn.sigmoid(crop_2)
    ###########################################################################################

    ###########################################################################################
    crop_3 = tf.keras.layers.ZeroPadding2D((2, 3))(crop_3)
    crop_3 = tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=(5,7),
                                    strides=1,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_3)
    crop_3 = tf.keras.layers.BatchNormalization()(crop_3)
    crop_3 = tf.keras.layers.LeakyReLU()(crop_3)

    crop_3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(crop_3)

    crop_3 = tf.keras.layers.ZeroPadding2D((1, 3))(crop_3)
    crop_3 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(5,7),
                                    strides=2,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_3)
    crop_3 = tf.keras.layers.BatchNormalization()(crop_3)
    crop_3 = tf.keras.layers.LeakyReLU()(crop_3)

    crop_3_max = tf.keras.layers.GlobalMaxPool2D()(crop_3)
    crop_3_max = tf.keras.layers.Dense(64/16)(crop_3_max)
    crop_3_max = tf.keras.layers.ReLU()(crop_3_max)
    crop_3_max = tf.keras.layers.Dense(64)(crop_3_max)

    crop_3_avg = tf.keras.layers.GlobalAveragePooling2D()(crop_3)
    crop_3_avg = tf.keras.layers.Dense(64/16)(crop_3_avg)
    crop_3_avg = tf.keras.layers.ReLU()(crop_3_avg)
    crop_3_avg = tf.keras.layers.Dense(64)(crop_3_avg)

    crop_3_sum = crop_3_max + crop_3_avg
    crop_3_sum = tf.nn.sigmoid(crop_3_sum)
    crop_3_sum = tf.expand_dims(crop_3_sum, 1)
    crop_3_sum = tf.expand_dims(crop_3_sum, 1)
    crop_3_ = tf.math.multiply(crop_3, crop_3_sum)

    crop_3_multi = tf.concat([ tf.reduce_max(crop_3_, -1, keepdims=True), tf.reduce_mean(crop_3_, -1, keepdims=True) ], -1)
    crop_3_multi = tf.keras.layers.Conv2D(filters=1,
                                    kernel_size=(5,7),
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_3_multi)
    crop_3_multi = tf.keras.layers.BatchNormalization()(crop_3_multi)
    crop_3 = tf.multiply(crop_3_multi, crop_3_)
    crop_3 = tf.nn.sigmoid(crop_3)
    ###########################################################################################
    
    crop = tf.concat([crop_1, crop_2, crop_3], 1)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)

    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h_max = tf.keras.layers.GlobalMaxPool2D()(h)
    h_max = tf.keras.layers.Dense(64/16)(h_max)
    h_max = tf.keras.layers.ReLU()(h_max)
    h_max = tf.keras.layers.Dense(64)(h_max)

    h_avg = tf.keras.layers.GlobalAveragePooling2D()(h)
    h_avg = tf.keras.layers.Dense(64/16)(h_avg)
    h_avg = tf.keras.layers.ReLU()(h_avg)
    h_avg = tf.keras.layers.Dense(64)(h_avg)

    h_sum = h_max + h_avg
    h_sum = tf.nn.sigmoid(h_sum)
    h_sum = tf.expand_dims(h_sum, 1)
    h_sum = tf.expand_dims(h_sum, 1)
    h_ = tf.math.multiply(h, h_sum)

    h_multi = tf.concat([ tf.reduce_max(h_, -1, keepdims=True), tf.reduce_mean(h_, -1, keepdims=True) ], -1)
    h_multi = tf.keras.layers.Conv2D(filters=1,
                                kernel_size=(5,7),
                                padding="same",
                                use_bias=False,
                                kernel_regularizer=l2(weight_decay))(h_multi)
    h_multi = tf.keras.layers.BatchNormalization()(h_multi)
    h = tf.multiply(h_multi, h_)
    h = tf.nn.sigmoid(h)

    h = tf.concat([h, crop], -1)

    h = tf.keras.layers.ZeroPadding2D((2,2))(h)
    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=5,
                               strides=2,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)

    h = tf.keras.layers.ZeroPadding2D((2,2))(h)
    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=5,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h_in = h
    ##################################################################################################  dual attention (position)
    h_position_att_B = tf.keras.layers.Conv2D(filters=128 // 8,
                                              kernel_size=1,
                                              strides=1,
                                              use_bias=False,
                                              kernel_regularizer=l2(weight_decay))(h)   # query 영향을 받는 feature
    h_position_att_B =  tf.keras.layers.Reshape((8*6, 128 // 8))(h_position_att_B)

    h_position_att_C = tf.keras.layers.Conv2D(filters=128 // 8,
                                              kernel_size=1,
                                              strides=1,
                                              use_bias=False,
                                              kernel_regularizer=l2(weight_decay))(h)   # key 영향을 주는 feature
    h_position_att_C = tf.keras.layers.Reshape((128 // 8, 8*6))(h_position_att_C)

    e = tf.matmul(h_position_att_B, h_position_att_C)   # 유사도 계산    (query와 key의 내적은 유사도를 측정하는것과 같다)
    attention = tf.nn.softmax(e, -1)
    attention = tf.reshape(attention, [-1, attention.shape[2], attention.shape[1]])


    h_position_att_D = tf.keras.layers.Conv2D(filters=128,
                                              kernel_size=1,
                                              strides=1,
                                              use_bias=False,
                                              kernel_regularizer=l2(weight_decay))(h)   # value 이 영향들에 대한 가중치
    h_position_att_D = tf.keras.layers.Reshape((128, 8*6))(h_position_att_D)
    h_position_out = tf.matmul(h_position_att_D, attention)
    h_position_out = tf.keras.layers.Reshape((8, 6, 128))(h_position_out)
    h_position_out = h_position_out + h
    ##################################################################################################

    ##################################################################################################
    h_channel_att_A = tf.keras.layers.Reshape((128, 8*6))(h)    # query
    h_channel_att_B = tf.keras.layers.Reshape((8*6, 128))(h)    # key
    e2 = tf.matmul(h_channel_att_A, h_channel_att_B)    # query 와 key의 유사도 계산
    e2 = tf.reduce_max(e2, -1, keepdims=True) - e2
    attention2 = tf.nn.softmax(e2, -1)

    value = tf.keras.layers.Reshape((128, 8*6))(h)

    h_channel_out = tf.matmul(attention2, value)
    h_channel_out = tf.keras.layers.Reshape((8, 6, 128))(h_channel_out)
    h_channel_out = h_channel_out + h
    ##################################################################################################

    h = h_channel_out + h_position_out  # [8. 6, 128]
    h = tf.multiply(h, tf.nn.softmax(h_in, -1))    # 이렇게하면 어떤효과가 있을지? --> 각 vector에 해당하는 좌료 값들에 대한 컨트라스트
    # self attention 준것을 입력에 h_in에 attention을 해줌

    h = tf.keras.layers.GlobalAveragePooling2D()(h)

    h = tf.keras.layers.Dense(1024)(h)
    h = tf.keras.layers.LeakyReLU()(h)
    h = tf.keras.layers.Dropout(0.5)(h)

    h = tf.keras.layers.Dense(1024)(h)
    h = tf.keras.layers.LeakyReLU()(h)
    h = tf.keras.layers.Dropout(0.5)(h)

    h = tf.keras.layers.Dense(num_classes)(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

model = fix_GL_network_2(num_classes=88)
model.summary()