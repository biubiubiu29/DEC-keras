import numpy as np
import pandas as pd

def extract_vgg16_features(x):
    from keras.preprocessing.image import img_to_array, array_to_img
    from keras.applications.vgg16 import preprocess_input, VGG16
    from keras.models import Model

    # im_h = x.shape[1]
    im_h = 224
    model = VGG16(include_top=True, weights='imagenet', input_shape=(im_h, im_h, 3))
    # if flatten:
    #     add_layer = Flatten()
    # else:
    #     add_layer = GlobalMaxPool2D()
    # feature_model = Model(model.input, add_layer(model.output))
    feature_model = Model(model.input, model.get_layer('fc1').output)
    print('extracting features...')
    x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((im_h,im_h))) for im in x])
    x = preprocess_input(x)  # data - 127. #data/255.#
    features = feature_model.predict(x)
    print('Features shape = ', features.shape)

    return features


def make_reuters_data(data_dir):
    np.random.seed(1234)
    from sklearn.feature_extraction.text import CountVectorizer
    from os.path import join
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        # did_to_cat = {k: did_to_cat[k] for k in list(did_to_cat.keys()) if len(did_to_cat[k]) > 1}
        for did in list(did_to_cat.keys()):
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open(join(data_dir, dat)) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did in did_to_cat:
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    print((len(data), 'and', len(did_to_cat)))
    assert len(data) == len(did_to_cat)

    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)

    from sklearn.feature_extraction.text import TfidfTransformer
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
    x = x[:10000].astype(np.float32)
    print(x.dtype, x.size)
    y = y[:10000]
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
    print('todense succeed')

    p = np.random.permutation(x.shape[0])
    x = x[p]
    y = y[p]
    print('permutation finished')

    assert x.shape[0] == y.shape[0]
    x = x.reshape((x.shape[0], -1))
    np.save(join(data_dir, 'reutersidf10k.npy'), {'data': x, 'label': y})


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    print('MNIST samples', x.shape)
    return x, y


def load_fashion_mnist():
    from keras.datasets import fashion_mnist  # this requires keras>=2.0.9
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    print('Fashion MNIST samples', x.shape)
    return x, y


def load_pendigits(data_path='./data/pendigits'):
    import os
    if not os.path.exists(data_path + '/pendigits.tra'):
        os.system('wget http://mlearn.ics.uci.edu/databases/pendigits/pendigits.tra -P %s' % data_path)
        os.system('wget http://mlearn.ics.uci.edu/databases/pendigits/pendigits.tes -P %s' % data_path)
        os.system('wget http://mlearn.ics.uci.edu/databases/pendigits/pendigits.names -P %s' % data_path)

    # load training data
    with open(data_path + '/pendigits.tra') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_train, labels_train = data[:, :-1], data[:, -1]
    print('data_train shape=', data_train.shape)

    # load testing data
    with open(data_path + '/pendigits.tes') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_test, labels_test = data[:, :-1], data[:, -1]
    print('data_test shape=', data_test.shape)

    x = np.concatenate((data_train, data_test)).astype('float32')
    y = np.concatenate((labels_train, labels_test))
    x /= 100.
    print('pendigits samples:', x.shape)
    return x, y


def load_usps(data_path='./data/usps'):
    import os
    if not os.path.exists(data_path+'/usps_train.jf'):
        if not os.path.exists(data_path+'/usps_train.jf.gz'):
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_train.jf.gz -P %s' % data_path)
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_test.jf.gz -P %s' % data_path)
        os.system('gunzip %s/usps_train.jf.gz' % data_path)
        os.system('gunzip %s/usps_test.jf.gz' % data_path)

    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64') / 2.
    y = np.concatenate((labels_train, labels_test))
    print('USPS samples', x.shape)
    return x, y


def load_reuters(data_path='./data/reuters'):
    import os
    if not os.path.exists(os.path.join(data_path, 'reutersidf10k.npy')):
        print('making reuters idf features')
        make_reuters_data(data_path)
        print(('reutersidf saved to ' + data_path))
    data = np.load(os.path.join(data_path, 'reutersidf10k.npy'),allow_pickle=True).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], -1)).astype('float64')
    y = y.reshape((y.size,))
    print(('REUTERSIDF10K samples', x.shape))
    return x, y


def load_retures_keras():
    from keras.preprocessing.text import Tokenizer
    from keras.datasets import reuters
    max_words = 1000

    print('Loading data...')
    (x, y), (_, _) = reuters.load_data(num_words=max_words, test_split=0.)
    print(len(x), 'train sequences')

    num_classes = np.max(y) + 1
    print(num_classes, 'classes')

    print('Vectorizing sequence data...')
    tokenizer = Tokenizer(num_words=max_words)
    x = tokenizer.sequences_to_matrix(x, mode='binary')
    print('x_train shape:', x.shape)

    return x.astype(float), y


def load_imdb():
    from keras.preprocessing.text import Tokenizer
    from keras.datasets import imdb
    max_words = 1000

    print('Loading data...')
    (x1, y1), (x2, y2) = imdb.load_data(num_words=max_words)
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    print(len(x), 'train sequences')

    num_classes = np.max(y) + 1
    print(num_classes, 'classes')

    print('Vectorizing sequence data...')
    tokenizer = Tokenizer(num_words=max_words)
    x = tokenizer.sequences_to_matrix(x, mode='binary')
    print('x_train shape:', x.shape)

    return x.astype(float), y


def load_newsgroups():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.datasets import fetch_20newsgroups
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    vectorizer = TfidfVectorizer(max_features=2000, dtype=np.float64, sublinear_tf=True)
    x_sparse = vectorizer.fit_transform(newsgroups.data)
    x = np.asarray(x_sparse.todense())
    y = newsgroups.target
    print('News group data shape ', x.shape)
    print("News group number of clusters: ", np.unique(y).size)
    return x, y


def load_cifar10(data_path='./data/cifar10'):
    from keras.datasets import cifar10
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y)).reshape((60000,))

    # if features are ready, return them
    import os.path
    if os.path.exists(data_path + '/cifar10_features.npy'):
        return np.load(data_path + '/cifar10_features.npy'), y

    # extract features
    features = np.zeros((60000, 4096))
    for i in range(6):
        idx = range(i*10000, (i+1)*10000)
        print("The %dth 10000 samples" % i)
        features[idx] = extract_vgg16_features(x[idx])

    # scale to [0,1]
    from sklearn.preprocessing import MinMaxScaler
    features = MinMaxScaler().fit_transform(features)

    # save features
    np.save(data_path + '/cifar10_features.npy', features)
    print('features saved to ' + data_path + '/cifar10_features.npy')

    return features, y


def load_stl(data_path='./data/stl'):
    import os
    assert os.path.exists(data_path + '/stl_features.npy') or not os.path.exists(data_path + '/train_X.bin'), \
        "No data! Use %s/get_data.sh to get data ready, then come back" % data_path

    # get labels
    y1 = np.fromfile(data_path + '/train_y.bin', dtype=np.uint8) - 1
    y2 = np.fromfile(data_path + '/test_y.bin', dtype=np.uint8) - 1
    y = np.concatenate((y1, y2))

    # if features are ready, return them
    if os.path.exists(data_path + '/stl_features.npy'):
        return np.load(data_path + '/stl_features.npy'), y

    # get data
    x1 = np.fromfile(data_path + '/train_X.bin', dtype=np.uint8)
    x1 = x1.reshape((int(x1.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
    x2 = np.fromfile(data_path + '/test_X.bin', dtype=np.uint8)
    x2 = x2.reshape((int(x2.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
    x = np.concatenate((x1, x2)).astype(float)

    # extract features
    features = extract_vgg16_features(x)

    # scale to [0,1]
    from sklearn.preprocessing import MinMaxScaler
    features = MinMaxScaler().fit_transform(features)

    # save features
    np.save(data_path + '/stl_features.npy', features)
    print('features saved to ' + data_path + '/stl_features.npy')

    return features, y
#------------------------------------------------
def getStopWords(datapath):
    stopwords = pd.read_csv(datapath, index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
    stopwords = stopwords["stopword"].values
    return stopwords

def jieba(string):
    import jieba
    seg_list = jieba.lcut(string, cut_all=False)
    stopwordsFile = r"./data/toutiao/stopwords.txt"
    stopwords = getStopWords(stopwordsFile)
    seg_list = filter(lambda x: len(x) > 1, seg_list)
    seg_list = filter(lambda x: x not in stopwords, seg_list)  # 去掉停用词
    return list(seg_list)


def get_word2vec():
    data_dir = './data/toutiao/'
    word2vec_path = 'data/toutiao/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
    print("正在加载预训练词向量……")
    from gensim.models import KeyedVectors
    wv_from_text = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
    if not exit(''):
        wv_from_text.save(data_dir+"w2v.model")
    wv_from_text.load(data_dir+"w2v.model")
    print("预训练词向量加载完毕。")
    return wv_from_text

def compute(text):
    if not text or len(text) == 0:
        return None
    value = [0] * 300
    count = 0
    wv_from_text = get_word2vec()  # 加载预训练词库
    for i in text:
        if i in wv_from_text:
            value = [a + b for a, b in zip(value, wv_from_text[i])]
            count += 1
    if count == 0:
        return None
    res = [round(k / count, 6) for k in value]
    return res

def make_toutiao_data(data_dir = './data/toutiao/'):

    data = pd.read_csv('data/toutiao/toutiao_cat_data.txt', sep='_!_',
                       names=['id', 'cat_index', 'cat', 'sentence', 'key'])
    data['sentence'] = data['sentence'].apply(jieba)  # 分词
    data['word2vec'] = data['sentence'].apply(compute)  # 计算句向量
    data_output = data[['cat_index', 'word2vec']]
    data_output = data_output.dropna(axis=0, how='any').reset_index()
    cat_to_cid = {101: 0, 103: 1, 104: 2, 107: 3, 108: 4}
    data_x = []
    target = []
    for i in range(len(data_output)):
        if int(data_output['cat_index'][i]) in cat_to_cid:
            data_x.append(data_output['word2vec'][i])
            target.append(cat_to_cid[data_output['cat_index'][i]])
    data_x = np.array(data_x)
    target = np.array(target)
    np.save(data_dir + 'toutiao_w2v.npy', {'data': data_x, 'label': target})

def load_toutiao(data_path = 'data/toutiao/'):
    import os
    if not os.path.exists(os.path.join(data_path, 'toutiao_w2v.npy')):
        print('making toutiao w2v features')
        make_toutiao_data(data_path)
        print(('reutersidf saved to ' + data_path))
    data = np.load(os.path.join(data_path, 'toutiao_w2v.npy'),allow_pickle=True).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x[:10000].astype(np.float32)
    print(x.dtype, x.size)
    y = y[:10000]
    print('count(0)=', np.sum(y == 0))
    print('count(1)=', np.sum(y == 1))
    print('count(2)=', np.sum(y == 2))
    print('count(3)=', np.sum(y == 3))
    print('count(4)=', np.sum(y == 4))
    x = x.reshape((x.shape[0], -1)).astype('float64')
    y = y.reshape((y.size,))
    print(('toutiao samples', x.shape))
    return x, y
#---------------------------------------------------------------
def load_data(dataset_name):
    if dataset_name == 'mnist':
        return load_mnist()
    elif dataset_name == 'fmnist':
        return load_fashion_mnist()
    elif dataset_name == 'usps':
        return load_usps()
    elif dataset_name == 'pendigits':
        return load_pendigits()
    elif dataset_name == 'reuters10k' or dataset_name == 'reuters':
        return load_reuters()
    elif dataset_name == 'stl':
        return load_stl()
    elif dataset_name == 'toutiao':
        return load_toutiao()
    else:
        print('Not defined for loading', dataset_name)
        exit(0)

if __name__=='__main__':
    load_toutiao(data_path='data/toutiao/')