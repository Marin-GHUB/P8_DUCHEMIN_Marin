# Standard Libraries
import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.cluster.vq import kmeans, vq
import random
import re
from time import time
from varname import argname2

# Plotting Libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import seaborn as sns

# Text Libraries
import nltk
import gensim
from gensim.models import Word2Vec
from wordcloud import WordCloud

# Image Libraries
import cv2 as cv

# Machine Learning Libraries
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint



##################### 
##################### Text Related Functions
##################### 

######### Getting all the words in a dataframe series
def find_all_words(dataframe, dataframe_serie_name):
    list_name = []
    for i in range(len(dataframe)):
        temp_list = dataframe.loc[i, dataframe_serie_name]
        for k in range(len(temp_list)):
            list_name.append(temp_list[k])
    list_name.sort()
    return list_name

######### Vectorizing Words
def create_W2V_X(model, data_df):
    X = []
    for index, description in data_df['description'].iteritems():
        X.append(sentence_vectorizer(description, model))
    X = np.asarray(X)
    return X

######### Vectorizing a sentence
def sentence_vectorizer(sentence, model):
    sent_vect = []
    numw = 0
    for word in sentence:
        try:
            if numw == 0:
                sent_vect = model.wv[word]
            else:
                sent_vect = np.add(sent_vect, model.wv[word])
            numw += 1
        except:
            pass
    return np.asarray(sent_vect) / numw

######### Creating a dataframe with the Bag of Words (BoW) for a given category
def create_BoW_dataframe(data_df, category):
    all_words = find_all_words(data_df, 'description')
    t0 = time()
    df = pd.DataFrame(columns=['words'])
    df['words'] = np.unique(np.array(all_words))
    cat_list = list(data_df[category].unique())
    for cat in cat_list:
        temp_df = data_df[data_df[category]==cat]
        temp_df.reset_index(inplace=True)
        cat_words = find_all_words(temp_df, 'description')
        word_counter = {}
        for word in cat_words:
            if word not in word_counter.keys():
                word_counter[word] = 1
            else:
                word_counter[word] += 1
        for key in word_counter:
            key_index = list(df[df['words']==key].index)[0]
            df.loc[key_index, cat] = word_counter[key]
    t1 = time()
    index_to_delete = []
    regex = re.compile("^\d+$")
    for i ,j in df['words'].iteritems():
        if len(j) == 1:
            index_to_delete.append(i)
        elif regex.match(j):
            index_to_delete.append(i)
    df.drop(index_to_delete, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print('Time needed for {} is {:.2f} seconds.'.format(category, t1-t0))
    return df

######### Going from the Part of Speech tags to the lemmatitizer attribute
def get_wordnet_pos(treebank_tag):

    """
    return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
    """
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        # As default pos in lemmatization is Noun
        return nltk.corpus.wordnet.NOUN
    
######### Function to lemmatize tagged tokens:
def lemmatize_tokens(tokens) :
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmated_test_tokens = []
    for word, tag in tokens:
        if tag is None:
            lemmatized_word = lemmatizer.lemmatize(word)
            lemmated_test_tokens.append(lemmatized_word)
        else:
            lemmatized_word = lemmatizer.lemmatize(word, pos=tag)
            lemmated_test_tokens.append(lemmatized_word)
    return lemmated_test_tokens

######### Text preprocessing
def text_preprocessing(text):
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # Tokenizing and lowering cases
    tokens = tokenizer.tokenize(text.lower())
    
    # Suppressing Stop Words
    clean_tokens = []
    for word in tokens:
        if word not in stop_words:
            clean_tokens.append(word)
            
    # Tagging the tokens
    pos_tagger = nltk.pos_tag(clean_tokens)
    tagged_tokens = list(map(lambda x: (x[0], get_wordnet_pos(x[1])), pos_tagger))
    
    # Lemmatization
    lemmated_tokens = []
    lemmatizer = nltk.stem.WordNetLemmatizer()
    for word, tag in tagged_tokens:
        if tag is None:
            lemmatized_word = lemmatizer.lemmatize(word)
            lemmated_tokens.append(lemmatized_word)
        else:
            lemmatized_word = lemmatizer.lemmatize(word, pos=tag)
            lemmated_tokens.append(lemmatized_word)
            
    return lemmated_tokens


##################### 
##################### Image Related Functions
##################### 

######### Automatic brightness and contrast optimization with optional histogram clipping
### Function form stackoverflow user nathancy 
### from https://stackoverflow.com/questions/57030125/automatically-adjusting-brightness-of-image-with-opencv
def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    # Calculate grayscale histogram
    hist = cv.calcHist([image],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result)

######### Showing the image with its histogram
def show_image_hist(image):
    fig, ax = plt.subplots(1, 2, figsize = (12,4))
    if len(image.shape) <3:
        ax[0].imshow(image, cmap='gray', vmin = 0, vmax = 255)
        n, bins, patches = ax[1].hist(image.flatten(), bins=range(256))
        ax[1].set_title('Histogram of the Number of Pixels per Intensity', fontsize=12)
        ax[1].set_xlabel('Intensity of the pixel')
        ax[1].set_ylabel('Number of pixels')
    else:
        ax[0].imshow(image)
        n, bins, patches = ax[1].hist(image.flatten(), bins=range(256))
        ax[1].set_title('Histogram of the Number of Pixels per Intensity', fontsize=12)
        ax[1].set_xlabel('Intensity of the pixel')
        ax[1].set_ylabel('Number of pixels')
        
######### Filtering the noise of an image
def filtering_image(image):
    # This argument will enable us to have an output image the same size of the input one
    ddepth = -1
    
    # Creating a kernel for a normalized box filter
    kernel = np.array(([[1,-1,1],
                        [-1,1,-1],
                        [1,-1,1]]), dtype=np.float32)
    
    # Apply the filter
    output_image = cv.filter2D(image, ddepth, kernel)
    
    return output_image

######### Image preprocessing
def image_preprocessing(df):
    path = 'Ressources/Images/' 
    for index, value in df['image'].iteritems():
        # Reading the image
        image = cv.imread(path+value)
        
        # Converting to grey scale
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        
        # Adjusting the brightness and contrast
        image = automatic_brightness_and_contrast(image)
        
        # Filtering the noise
        image = filtering_image(image)
        
        # Saving the image 
        cv.imwrite(path+'Preprocessed/'+value, image)
        
        # indexing new path
        df.loc[index, 'image'] = path+'Preprocessed/'+value
    return df

######### Creating descriptors of an image
def create_descriptors(image):
    orb = cv.ORB_create(nfeatures=1200)
    kp, des = orb.detectAndCompute(image, None)
    return kp, des

######### Creating descriptors for all images
def fill_descriptors(df):
    all_images = []
    descriptors_list = []
    for i, path in df['image'].iteritems():
        image = cv.imread(path)
        kp, des = create_descriptors(image)
        descriptors_list.append((path, des))
        all_images.append(image)
    return descriptors_list, all_images

######### Showing the cluster histogram for the example image
def build_histogram(kmeans, des, voc):
    res = kmeans.predict(des)
    hist = np.zeros(len(voc))
    nb_des = len(des)
    for i in res:
        hist[i] += 1.0/nb_des
    plt.figure(figsize = ((12,12)))           
    plt.bar(np.arange(len(hist)), hist)
    plt.title('Cluster Histogram for the Test Image', fontsize=12)
    plt.xlabel('Clusters')
    plt.ylabel('Descriptors Number')
    plt.show()
    plt.savefig('Ressources' + '/' + 'Soutenance' + '/' + 'test_image_hist.png')
    return hist


##################### 
##################### Classifier Related Functions
##################### 

######### Separating the data into train and test
def train_test_separation(X, y, encoding='OneHot'):
    if encoding == 'OneHot':
        encoder = OneHotEncoder(handle_unknown='ignore')
    elif encoding == 'Label':
        encoder = LabelEncoder()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = 0)
    if encoding == 'OneHot':
        y_train_encoded = encoder.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
        y_test_encoded = encoder.fit_transform(np.array(y_test).reshape(-1, 1)).toarray()
    elif encoding == 'Label':
        y_train_encoded = encoder.fit_transform(y_train)
        y_test_encoded = encoder.fit_transform(y_test)
        
    set_dict = {'X_train' : X_train, 
                'X_test' : X_test,
                'y_train' : y_train,
                'y_test' : y_test,
                'y_train_encoded' : y_train_encoded,
                'y_test_encoded' : y_test_encoded}
    return set_dict

######### Plotting a confusion matrix for the classifiers
def create_confusion_matrix(labels_test, labels_preds, labels_true, classifier_name):
    lbl_to_plot = set(labels_true)
    lenght = len(lbl_to_plot)
    if lenght == 7:
        lvl_nb = 1
    elif lenght == 25:
        lvl_nb = 2
    else :
        lvl_nb = 3
    conf_mat = confusion_matrix(labels_test, labels_preds)
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
                xticklabels=lbl_to_plot, 
                yticklabels=lbl_to_plot)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("CONFUSION MATRIX - {} Classifier on the Categories of Level {}\n"
              .format(classifier_name, lvl_nb), size=16)
    plt.savefig('Ressources' + '/' + 'Soutenance' + '/' + 'Confusion Matrix {} Level {}.png'
               .format(classifier_name, lvl_nb))

######### Plotting the loss and accuracy of Neural Networks
def show_loss_acc(model_history, name):
    fig, ax = plt.subplots(1, 2, figsize=(24,12))
    ax[0].set_title('Loss per Number of Epochs', fontsize=12)
    ax[0].set_xlabel('Number of Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].plot(model_history.history['loss'], label='train')
    ax[0].plot(model_history.history['val_loss'], label='test')
    ax[0].legend(fontsize=8)
    ax[1].set_title('Accuracy per Number of Epochs', fontsize=12)
    ax[1].set_xlabel('Number of Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].plot(model_history.history['accuracy'], label='train')
    ax[1].plot(model_history.history['val_accuracy'], label='test')
    ax[1].legend(fontsize=8)
    plt.savefig('Ressources' + '/' + 'Soutenance' + '/' + 'Loss_Acc_{}.png'.format(name))
    
######### Classifying with 'classic' classifiers (MNB, Random Forest)
def classic_classifier(classifier, method_set, classifier_name, labels):
    # We train the classifier
    t0 = time()
    classifier.fit(method_set['X_train'], method_set['y_train'])

    # We try to predict the test set
    classifier_y_pred = classifier.predict(method_set['X_test'])
    t1 = time()
    classifier_time = t1-t0

    # We evaluate the accuracy of the model
    classifier_accuracy = metrics.accuracy_score(method_set['y_test'], classifier_y_pred)*100
    print('Our {} has an accuracy of : {:.2f}% in {:.2f}s.'
          .format(name, classifier_accuracy, classifier_time))

    # We create a confusion matrix to see where are the most problematic categories
    create_confusion_matrix(method_set['y_test'], classifier_y_pred, labels, classifier_name)
    return classifier_accuracy, classifier_time

######### Classifying images with a 'classic' classifier
def SVC_Classifier(method_set, classifier_name):
    SVC_model = LinearSVC(max_iter=60000)
    # We train the classifier
    t0 = time()
    classifier = SVC_model.fit(method_set['X_train'], method_set['y_train_encoded'])

    # We try to predict the test set
    classifier_y_pred = classifier.predict(method_set['X_test'])
    t1 = time()
    classifier_time = t1-t0

    # We evaluate the accuracy of the model
    classifier_accuracy = metrics.accuracy_score(method_set['y_test_encoded'], classifier_y_pred)*100
    print('Our Linear SVC Classifier has an accuracy of : {:.2f}% in {:.2f}s.'
          .format(classifier_accuracy, classifier_time))

    # We create a confusion matrix to see where are the most problematic categories
    create_confusion_matrix(method_set['y_test_encoded'], classifier_y_pred, method_set['y_test_encoded'], classifier_name)
    return classifier_accuracy, classifier_time 

######### Classifying with RNN classifiers
def RNN_classifier(method_set, labels, W2V):
    # Counting labels
    category_label = set(labels)
    label_number = len(category_label)
    parameters_number = 4 *((method_set['X_train'].shape[1]+1) * label_number + label_number^2)

    # We create the RNN
    RNN_model = keras.Sequential()
    RNN_model.add(layers.Flatten())
    RNN_model.add(layers.Dense(parameters_number, activation='ReLU'))
    RNN_model.add(layers.Dense(label_number, activation='softmax'))
    RNN_model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    
    # We train the RNN to classify
    t0 = time()

    epochs = 240
    batch_size = 64

    RNN_history = RNN_model.fit(method_set['X_train'], method_set['y_train_encoded'], 
                                    epochs=epochs, batch_size=batch_size,
                                    validation_split=0.1, verbose=0)

    # We try to predict the test set
    RNN_y_pred = RNN_model.predict(method_set['X_test'])
    t1 = time()
    RNN_model_time = t1-t0

    # We evaluate the accuracy of the model
    rounded_labels = np.argmax(method_set['y_test_encoded'], axis=1)
    rounded_pred = np.argmax(RNN_y_pred, axis=1)
    RNN_accuracy = RNN_history.history['val_accuracy'][-1]*100
    print('Our Recurrent Neural Network has an accuracy of : {:.2f}% in {:.2f}s.'
          .format(RNN_accuracy, RNN_model_time))

    # We look at the evolutions of the loss and accuracy over epoch
    show_loss_acc(RNN_history, 'RNN {} Level 1'.format(W2V))

    # We look at the confusion matrix
    create_confusion_matrix(rounded_labels, rounded_pred, labels, 'RNN {}'.format(W2V))
    
    return RNN_accuracy, RNN_model_time

######### Classifying with a CNN classifier
def CNN_classifier(labels, all_images):
    # Normalizing the sizes of the images
    resized_all_images = []
    for image in all_images:
        resized_image = cv.resize(image, (100, 100),interpolation = cv.INTER_NEAREST)
        resized_all_images.append(resized_image)
    
    # Split Data into Train and Test
    CNN_X = np.asarray(resized_all_images)
    encoder = OneHotEncoder()
    CNN_y = encoder.fit_transform(np.array(labels).reshape(-1, 1)).toarray()
    CNN_X_train, CNN_X_test, CNN_y_train, CNN_y_test = train_test_split(CNN_X, CNN_y,
                                                                    test_size=0.20, random_state = 0)
    
    # Counting labels
    category_label = set(labels)
    label_number = len(category_label)
        
    # Constructing the Convolutional Neural Network
    # Adding some layers
    CNN_model = keras.Sequential()
    CNN_model.add(layers.Conv2D(filters=48, kernel_size=3,
                                activation='relu',input_shape=[100, 100, 3]))
    CNN_model.add(layers.MaxPool2D(pool_size=2, strides=2))
    CNN_model.add(layers.Dropout(0.5))
    CNN_model.add(layers.Conv2D(filters=48, kernel_size=3,
                                activation='relu',input_shape=[100, 100, 3]))
    CNN_model.add(layers.MaxPool2D(pool_size=2, strides=2))
    CNN_model.add(layers.Dropout(0.2))
    CNN_model.add(layers.Flatten())
    CNN_model.add(layers.Dense(128, activation='relu'))
    CNN_model.add(layers.Dense(64, activation='relu'))
    CNN_model.add(layers.Dense(label_number, activation='softmax'))
    
    # Compiling the CNN
    CNN_model.compile(loss='categorical_crossentropy',optimizer='adam',
                  metrics=['accuracy'])

    # Fitting the CNN 
    t0 = time()
    CNN_history = CNN_model.fit(CNN_X_train, CNN_y_train,
                                          batch_size = 128, epochs=36, 
                                          validation_split=0.2,verbose=0,
                                          shuffle=True)
    
    # We try to predict
    CNN_y_pred = CNN_model.predict(CNN_X_test)
    t1 = time()
    CNN_model_time = t1-t0
    
    # We evaluate the accuracy of the model
    rounded_labels = np.argmax(CNN_y_test, axis=1)
    rounded_pred = np.argmax(CNN_y_pred, axis=1)
    CNN_accuracy = CNN_history.history['val_accuracy'][-1]*100
    print('Our Convolutional Neural Network has an accuracy of : {:.2f}% in {:.2f}s.'
          .format(CNN_accuracy, CNN_model_time))

    # We look at the evolutions of the loss and accuracy over epoch
    show_loss_acc(CNN_history, 'RNN {} Level 1'.format('Image'))

    # We look at the confusion matrix
    create_confusion_matrix(rounded_labels, rounded_pred, labels, 'CNN {}'.format('Image'))
    
    return CNN_accuracy, CNN_model_time

######### Function to compile the text results in a table
def create_text_results_table(text_results_dict):
    text_results_df = pd.DataFrame(
        {'Accuracy':[text_results_dict['MNB_lvl_1'][0], text_results_dict['RF_lvl_1'][0],
                     text_results_dict['CBOWRNN_lvl_1'][0], text_results_dict['SGRNN_lvl_1'][0], 
                     text_results_dict['MNB_lvl_2'][0], text_results_dict['RF_lvl_2'][0], 
                     text_results_dict['CBOWRNN_lvl_2'][0], text_results_dict['SGRNN_lvl_2'][0], 
                     text_results_dict['MNB_lvl_3'][0], text_results_dict['RF_lvl_3'][0], 
                     text_results_dict['CBOWRNN_lvl_3'][0], text_results_dict['SGRNN_lvl_3'][0]],
         'Time':[text_results_dict['MNB_lvl_1'][1], text_results_dict['RF_lvl_1'][1],
                     text_results_dict['CBOWRNN_lvl_1'][1], text_results_dict['SGRNN_lvl_1'][1], 
                     text_results_dict['MNB_lvl_2'][1], text_results_dict['RF_lvl_2'][1], 
                     text_results_dict['CBOWRNN_lvl_2'][1], text_results_dict['SGRNN_lvl_2'][1], 
                     text_results_dict['MNB_lvl_3'][1], text_results_dict['RF_lvl_3'][1], 
                     text_results_dict['CBOWRNN_lvl_3'][1], text_results_dict['SGRNN_lvl_3'][1]]},
        index = pd.MultiIndex.from_tuples(
            [('Level 1', 'Multinomial Naive Bayes'),('Level 1','Random Forest'),
             ('Level 1', 'RNN on Continuous Bag of Word'),('Level 1','RNN on Skip Gram'),
             ('Level 2', 'Multinomial Naive Bayes'),('Level 2','Random Forest'),
             ('Level 2', 'RNN on Continuous Bag of Word'),('Level 2','RNN on Skip Gram'),
             ('Level 3', 'Multinomial Naive Bayes'),('Level 3','Random Forest'),
             ('Level 3', 'RNN on Continuous Bag of Word'),('Level 3','RNN on Skip Gram')],
            names = ['Level of Category', 'Model']))
    return text_results_df

######### Function to compile the image results in a table
def create_image_results_table(image_results_dict):
    image_results_df = pd.DataFrame(
        {'Accuracy':[image_results_dict['SVC_lvl_1'][0], image_results_dict['CNN_lvl_1'][0],
                     image_results_dict['SVC_lvl_2'][0], image_results_dict['CNN_lvl_2'][0], 
                     image_results_dict['SVC_lvl_3'][0], image_results_dict['CNN_lvl_3'][0]],
         'Time':[image_results_dict['SVC_lvl_1'][1], image_results_dict['CNN_lvl_1'][1],
                     image_results_dict['SVC_lvl_2'][1], image_results_dict['CNN_lvl_2'][1], 
                     image_results_dict['SVC_lvl_3'][1], image_results_dict['CNN_lvl_3'][1]]},
                     index = pd.MultiIndex.from_tuples(
            [('Level 1', 'Linear SVC'),('Level 1','CNN'),
             ('Level 2', 'Linear SVC'),('Level 2','CNN'),
             ('Level 3', 'Linear SVC'),('Level 3','CNN')],
            names = ['Level of Category', 'Model']))
    return image_results_df


##################### 
##################### Other Functions
##################### 

######### Showing wordcloud of dataframe
def create_WC_from_DF(*dataframe):
    name = argname2('*dataframe')
    for name, dataframe in zip(name, dataframe):
        dataframe.__dfname__ = name
    t0 = time()
    col_list = list(dataframe.columns)
    col_list.pop(0)
    # Finding the right size for the subplot
    total_dimension = len(col_list)
    sqrt = np.sqrt(total_dimension)
    test_dim = total_dimension/int(sqrt)
    if test_dim == int(test_dim):
        dim_1, dim_2 = int(sqrt), int(test_dim)
    else:
        dim_1, dim_2 = int(sqrt), int(test_dim+1)
    fig, ax = plt.subplots(dim_1, dim_2, figsize = (dim_2*6,dim_2*6))
    # Going through the columns to create wordclouds
    for i in range(len(col_list)):
        # Wordcloud creation
        text_list = []
        for j, k in dataframe[col_list[i]].iteritems():
            if np.isnan(k) not in [True]:
                word = dataframe.loc[j, 'words']
                word_list = [word]*int(k)
                text_list.extend(word_list)
        text = ' '.join(text_list)
        wordcloud = WordCloud(width=800, height=800, collocations=False).generate(text)
        # Wordcloud plotting
        l = i%dim_1
        m = i//dim_1
        ax[l,m].imshow(wordcloud)
        ax[l,m].set_title('Category {}:'.format(col_list[i]),
                          fontsize = 12, fontweight = 'bold')
    t1 = time()
    print('Time elapsed : {} s.\n'.format(t1-t0))
    plt.savefig('Ressources' + '/' + 'Soutenance' + '/' + 'WordCloud_{}.png'.format(dataframe.__dfname__))
    plt.show()

######### Transforming through t-SNE
def tsne_transformation(X, name, tsne):
    t0 = time()
    X_tsne = tsne.fit_transform(X)
    t1 = time()
    tsne_time = t1-t0
    print('Time elapsed for the {} method: {} s.'.format(name, tsne_time))
    return X_tsne

######### Plotting the t-SNE
def visualize_tsne(X_method, labels, method_name, labels_lvl, text_image):
    fig, ax = plt.subplots(figsize = (24,12))
    color_palette = [mcolors.CSS4_COLORS[i] for i in mcolors.CSS4_COLORS]
    ax.set_prop_cycle(color=color_palette[::-5])

    groups = pd.DataFrame(X_method, columns=['x', 'y']).assign(category=labels).groupby('category')
    for name, points in groups:
        ax.scatter(points.x, points.y, label=name)

    ax.set_title('Visual Distribution of Categories')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')
    ax.legend(bbox_to_anchor=(0.9,0), ncol=7)

    plt.savefig('Ressources' + '/' + 'Soutenance' + '/' + '{} tSNE {} for Categories of Level {}.png'
                .format(text_image, method_name, labels_lvl))
    plt.show()