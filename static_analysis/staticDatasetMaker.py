import csv
import os
#import statistics
from gensim.models import word2vec
from gensim.models import FastText
import logging


# make BSY_all_data.txt for train w2v & fasttext model
def all_dataMaker(benign_cnt, malware_cnt, label_file):
    # make all_data.txt file
    all_data_element = open('./dataset/1_all_data.txt', 'a')
    csv_data = open(label_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_data)

    for line in csv_reader:
        try:
            # 두개 다 16081개로 1:1 비율이 맞으면 끝내기!
            if benign_cnt == 16081 and malware_cnt == 16081:
                print("malware_cnt = %d     benign_cnt = %d " % (malware_cnt, benign_cnt))
                break

            # benign 갯수랑 malware 갯수 16081 되면 안돌고 pass
            if line[1] == '0' and benign_cnt == 16081:
                continue
            elif line[1] == '1' and malware_cnt == 16081:
                continue

            # open opcode data
            op_file = open('./dataset/opcode/%s.txt' % line[0], 'r')
            all_data_element.write(str(line[1]) + '##')

            i = 0
            # make contents of opcode part of all data
            for opline in op_file:
                if i >= 500:
                    break
                # cutting over opcode mean
                if opline.find('?') != -1:
                    opline.replace('?', '')
                    all_data_element.write(opline.strip() + ' ')
                    i += 1
                else:
                    all_data_element.write(opline.strip() + ' ')
                    i += 1

            all_data_element.write('\n')

            op_file.close()

            if line[1] == '0':
                benign_cnt += 1
            elif line[1] == '1':
                malware_cnt += 1
        except:
            pass
    csv_data.close()

    return benign_cnt, malware_cnt

# remove label at all_data text file
def label_del():
    train_label = open('./dataset/1_all_data.txt', 'r')
    save = open('./data/train/1_train_delet_label.txt', 'w')
    for line in train_label:
        train_list = []
        label, contents = line.split('##', 1)
        train_list.append(contents)
        train_list = ''.join(train_list)
        print(train_list, file=save, end='')
    del train_list

def train_word2vec():
    if not os.path.exists('./data/train/1_train_delet_label.txt'):
        label_del()
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus('./data/train/1_train_delet_label.txt')

    # start word2vec train
    print("word2vec 300 train start!")
    model = word2vec.Word2Vec(sentences, iter=50, min_count=1, size=300, workers=4, window=5, sg=1)
    model.save('./summary/train/1_word2vec_model_300')
    print("Done train_word2vec 300")

    print("word2vec 600 train start!")
    model = word2vec.Word2Vec(sentences, iter=50, min_count=1, size=600, workers=4, window=5, sg=1)
    model.save('./summary/train/1_word2vec_model_600')
    print("Done train_word2vec 600")

    print("word2vec 900 train start!")
    model = word2vec.Word2Vec(sentences, iter=50, min_count=1, size=900, workers=4, window=5, sg=1)
    model.save('./summary/train/1_word2vec_model_900')
    print("Done train_word2vec 900")

def train_fasttext():
    if not os.path.exists('./data/train/1_train_delet_label.txt'):
        label_del()
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus('./data/train/1_train_delet_label.txt')

    # start fasttext train
    print("fasttext 300 train start!")
    model2 = FastText(sentences, iter=50, min_count=1, size=300, workers=4, min_n=2, max_n=6)
    model2.save('./summary/train/1_fasttext_model_300')
    print("Done train_fasttext 300")

    print("fasttext 600 train start!")
    model2 = FastText(sentences, iter=50, min_count=1, size=600, workers=4, min_n=2, max_n=6)
    model2.save('./summary/train/1_fasttext_model_600')
    print("Done train_fasttext 600")

    print("fasttext 900 train start!")
    model2 = FastText(sentences, iter=50, min_count=1, size=900, workers=4, min_n=2, max_n=6)
    model2.save('./summary/train/1_fasttext_model_900')
    print("Done train_fasttext 900")


# build vocab.txt file
def build_vocab(model, vocab_dir):
    words = []
    # for i in range(len(model.wv.vocab)):
    for i in range(len(model.wv.vocab)):
        words.append(model.wv.index2word[i])

    # Add <pad> to word_vocab
    words = ['<PAD>'] + list(words)

    open(vocab_dir, mode='w', encoding='utf-8', errors='ignore').write('\n'.join(words) + '\n')
'''
def MS_dataset_Maker(label_file):
    # make all_data.txt file
    all_data_element = open('./dataset/BSY_all_data_seq600.txt', 'a')
    csv_data = open(label_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_data)
    for line in csv_reader:
        try:
            # open opcode data
            op_file = open('./dataset/MS_dataset_opcode/%s.txt' % line[0], 'r')
            all_data_element.write(str(line[1]) + '##')

            # make contents of opcode part of all data
            i = 0
            for opline_char in op_file:
                # cutting over opcode mean
                if i >= 550:
                    break
                opline = opline_char.replace(' ', '')
                if opline.find('?') != -1:
                    opline.replace('?', '')
                    all_data_element.write(opline.strip() + ' ')
                    i += 1
                else:
                    all_data_element.write(opline.strip() + ' ')
                    i += 1


            op_file.close()

        except:
            pass

        try:
            # write API contents at all_data.txt
            API_file = open('./dataset/MS_dataset_API/%s.txt' % line[0], 'r')
            k = 0
            # api 들을 list에 저장하여 all_data.txt에 넣을 contents 만들기
            for APIline in API_file:
                # cutting over API mean
                if k >= 50:
                    break
                if APIline.find('?') != -1 or APIline.find('@') != -1 or APIline.find('_') != -1 or APIline.find('$') != -1 or APIline.find('(') != -1 or APIline.find(')') != -1:
                    pass
                else:
                    all_data_element.write(APIline.strip() + ' ')
                    k += 1


            all_data_element.write('\n')
            API_file.close()
        except:
            pass

    csv_data.close()


# make words name & vector dictionary
def make_vectorDict(model, vocab_dir, embedding_size):
    vector_list = []
    name_list = []
    with open(vocab_dir, mode='r', encoding='utf-8', errors='ignore') as fp:
        words = [_.strip() for _ in fp.readlines()]

    for i in range(len(words)):
        if words[i] in model.wv.vocab:
            name_list.append(words[i])
            embedding_vector = model.wv[words[i]]
            vector_list.append(sum(embedding_vector) / embedding_size)

    data_dict = dict(zip(name_list, vector_list))

    return data_dict

def vector_maker(model, vocab_data, all_vector_txt, label_file, embedding_size):
    data_dict = make_vectorDict(model, vocab_data, embedding_size)

    # make all_data_vector.txt file
    all_data_element = open(all_vector_txt, 'a')
    csv_data = open(label_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_data)
    for line in csv_reader:
        try:
            # open opcode data
            file = open('./dataset/thinkingmind/pre_opcode/%s.txt' % line[0], 'r')
            all_data_element.write(str(line[1]) + '##')

            # make contents of opcode part of all data
            i = 0
            for opline_char in file:
                # cutting over opcode mean
                if i >= 900:
                    break

                opline = opline_char.replace(' ', '')

                if opline.find('?') != -1:
                    opline.replace('?', '')
                    opcode_input = data_dict[opline.strip()]
                    all_data_element.write('%f ' % opcode_input)
                    i += 1
                else:
                    opcode_input = data_dict[opline.strip()]
                    all_data_element.write('%f ' % opcode_input)
                    i += 1

            file.close()
        except:
            pass

        try:
            # write API contents at all_data.txt
            file = open('./dataset/thinkingmind/pre_api/%s.txt' % line[0], 'r')
            k = 0
            # api 들을 list에 저장하여 all_data.txt에 넣을 contents 만들기
            for APIline in file:
                # cutting over API mean
                if k >= 100:
                    break
                API_input = data_dict[APIline.strip()]
                all_data_element.write('%f ' % API_input)
                k += 1

            all_data_element.write('\n')
            file.close()
        except:
            pass

    csv_data.close()

def make_vector_dataset(all_vector_txt, embedding_file_name):
    tempo = []
    with open(all_vector_txt, 'r') as all_data:
        for line in all_data:
            if len(line) > 5:
                tempo.append(line)
    size = len(tempo)//100
    train = tempo[0:80*size]
    valid = tempo[80*size:90*size]
    test = tempo[90*size:100*size]
    save = open('./data/train/BSY_train_%s.txt' % embedding_file_name, 'w')
    for line in train:
        print(line, file=save)

    save = open('./data/train/BSY_valid_%s.txt' % embedding_file_name, 'w')
    for line in valid:
        print(line, file=save)

    save = open('./data/train/BSY_test_%s.txt' % embedding_file_name, 'w')
    for line in test:
        print(line, file=save)
'''

'''
def make_dataset():
    tempo = []
    with open('./dataset/BSY_all_data.txt', 'r') as all_data:
        for line in all_data:
            if len(line) > 5:
                tempo.append(line)
    size = len(tempo)//100
    train = tempo[0:80*size]
    valid = tempo[80*size:90*size]
    test = tempo[90*size:100*size]
    save = open('./data/train/BSY_train.txt', 'w')
    for line in train:
        print(line, file=save)

    save = open('./data/train/BSY_valid.txt', 'w')
    for line in valid:
        print(line, file=save)

    save = open('./data/train/BSY_test.txt', 'w')
    for line in test:
        print(line, file=save)
'''

def checker():
    print("checker 실행...")
    mal_cnt = 0
    ben_cnt = 0
    #opcode_path = './dataset/opcode/'
    #file_list = os.listdir(opcode_path)
    with open('./dataset/1_all_data.txt', 'r') as all_data:
        for line in all_data:
            label = line[0]
            #print(label)
            if label == '0':
                ben_cnt += 1
            elif label == '1':
                mal_cnt += 1
    print("checker... malware count : " + str(mal_cnt))
    print("checker... benign count : " + str(ben_cnt))



def make_dataset():
    mal_cnt = 0
    ben_cnt = 0
    train_file = open('./data/train/1_train.txt', 'w')
    valid_file = open('./data/train/1_valid.txt', 'w')
    test_file = open('./data/train/1_test.txt', 'w')
    with open('./dataset/1_all_data.txt', 'r') as all_data:
        for line in all_data:
            label, _ = line.split('##', 1)
            if label == '0':
                ben_cnt += 1
                if ben_cnt <= 12864:
                    print(line, file=train_file)
                elif ben_cnt > 12864 and ben_cnt <= 14473:
                    print(line, file=valid_file)
                else:
                    print(line, file=test_file)
            elif label == '1':
                mal_cnt += 1
                if mal_cnt <= 12864:
                    print(line, file=train_file)
                elif mal_cnt > 12864 and mal_cnt <= 14473:
                    print(line, file=valid_file)
                else:
                    print(line, file=test_file)

    print("Finish make_dataset... malware cnt : " + str(mal_cnt))
    print("Finish make_dataset... benign cnt : " + str(ben_cnt))


def main():
    benign_cnt = 0
    malware_cnt = 0

    train_label = './data/label/trainSet.csv'
    pre_label = './data/label/preSet.csv'
    final1_label = './data/label/finalSet1.csv'
    final2_label = './data/label/finalSet2.csv'
    MS_label = './data/label/MSSet1.csv'

    word_vocab = './data/train/1_word_vocab.txt'

    w2v300_model_dir = './summary/train/1_word2vec_model_300'
    w2v600_model_dir = './summary/train/1_word2vec_model_600'
    w2v900_model_dir = './summary/train/1_word2vec_model_900'
    fast300_model_dir = './summary/train/1_fasttext_model_300'
    fast600_model_dir = './summary/train/1_fasttext_model_600'
    fast900_model_dir = './summary/train/1_fasttext_model_900'

    # word2vec 300 dataset maker
    # MS_dataset_Maker(MS_label)
    # all_dataMaker(MS_label)
    #print("start all_dataMaker...")
    #benign_cnt, malware_cnt = all_dataMaker(benign_cnt, malware_cnt, train_label)
    #print("하나 끝! malware_cnt = %d     benign_Cnt = %d " % (malware_cnt, benign_cnt))
    #benign_cnt, malware_cnt = all_dataMaker(benign_cnt, malware_cnt, pre_label)
    #print("둘 끝! malware_cnt = %d     benign_Cnt = %d " % (malware_cnt, benign_cnt))
    #benign_cnt, malware_cnt = all_dataMaker(benign_cnt, malware_cnt, final1_label)
    #print("셋 끝! malware_cnt = %d     benign_Cnt = %d " % (malware_cnt, benign_cnt))
    #benign_cnt, malware_cnt = all_dataMaker(benign_cnt, malware_cnt, final2_label)
    #print("넷 끝! malware_cnt = %d     benign_Cnt = %d " % (malware_cnt, benign_cnt))
    #print("finish all_dataMaker...")

    checker()

    # train embedding file
    train_word2vec()
    w2v300_model = word2vec.Word2Vec.load(w2v300_model_dir)
    build_vocab(w2v300_model, word_vocab)
    train_fasttext()

    make_dataset()



#checker()
main()

'''
all_vector_w2v300 = './dataset/BSY_all_vector_w2v300.txt'
all_vector_w2v600 = './dataset/BSY_all_vector_w2v600.txt'
all_vector_w2v900 = './dataset/BSY_all_vector_w2v900.txt'
all_vector_fast300 = './dataset/BSY_all_vector_fast300.txt'
all_vector_fast600 = './dataset/BSY_all_vector_fast600.txt'
all_vector_fast900 = './dataset/BSY_all_vector_fast900.txt'
'''
'''
print("start make word2vec 300 vector dataset")


vector_maker(w2v300_model, word_vocab, all_vector_w2v300, MS_label, embedding_size=300)
vector_maker(w2v300_model, word_vocab, all_vector_w2v300, train_label, embedding_size=300)
vector_maker(w2v300_model, word_vocab, all_vector_w2v300, pre_label, embedding_size=300)
vector_maker(w2v300_model, word_vocab, all_vector_w2v300, final1_label, embedding_size=300)
vector_maker(w2v300_model, word_vocab, all_vector_w2v300, final2_label, embedding_size=300)
make_vector_dataset(all_vector_w2v300, embedding_file_name='word2vec300')
print("finish make word2vec 300 vector dataset")

print("start make word2vec 600 vector dataset")
w2v600_model = word2vec.Word2Vec.load(w2v600_model_dir)
vector_maker(w2v600_model, word_vocab, all_vector_w2v600, MS_label, embedding_size=600)
vector_maker(w2v600_model, word_vocab, all_vector_w2v600, train_label, embedding_size=600)
vector_maker(w2v600_model, word_vocab, all_vector_w2v600, pre_label, embedding_size=600)
vector_maker(w2v600_model, word_vocab, all_vector_w2v600, final1_label, embedding_size=600)
vector_maker(w2v600_model, word_vocab, all_vector_w2v600, final2_label, embedding_size=600)
make_vector_dataset(all_vector_w2v600, embedding_file_name='word2vec600')
print("finish make word2vec 600 vector dataset")

print("start make word2vec 900 vector dataset")
w2v900_model = word2vec.Word2Vec.load(w2v900_model_dir)
vector_maker(w2v900_model, word_vocab, all_vector_w2v900, MS_label, embedding_size=900)
vector_maker(w2v900_model, word_vocab, all_vector_w2v900, train_label, embedding_size=900)
vector_maker(w2v900_model, word_vocab, all_vector_w2v900, pre_label, embedding_size=900)
vector_maker(w2v900_model, word_vocab, all_vector_w2v900, final1_label, embedding_size=900)
vector_maker(w2v900_model, word_vocab, all_vector_w2v900, final2_label, embedding_size=900)
make_vector_dataset(all_vector_w2v900, embedding_file_name='word2vec900')
print("finish make word2vec 900 vector dataset")

print("start make fasttext 300 vector dataset")
fast300_model = FastText.load(fast300_model_dir)
vector_maker(fast300_model, word_vocab, all_vector_fast300, MS_label, embedding_size=300)
vector_maker(fast300_model, word_vocab, all_vector_fast300, train_label, embedding_size=300)
vector_maker(fast300_model, word_vocab, all_vector_fast300, pre_label, embedding_size=300)
vector_maker(fast300_model, word_vocab, all_vector_fast300, final1_label, embedding_size=300)
vector_maker(fast300_model, word_vocab, all_vector_fast300, final2_label, embedding_size=300)
make_vector_dataset(all_vector_fast300, embedding_file_name='fasttext300')
print("finish make fasttext 300 vector dataset")

print("start make fasttext 600 vector dataset")
fast600_model = FastText.load(fast600_model_dir)
vector_maker(fast600_model, word_vocab, all_vector_fast600, MS_label, embedding_size=600)
vector_maker(fast600_model, word_vocab, all_vector_fast600, train_label, embedding_size=600)
vector_maker(fast600_model, word_vocab, all_vector_fast600, pre_label, embedding_size=600)
vector_maker(fast600_model, word_vocab, all_vector_fast600, final1_label, embedding_size=600)
vector_maker(fast600_model, word_vocab, all_vector_fast600, final2_label, embedding_size=600)
make_vector_dataset(all_vector_fast600, embedding_file_name='fasttext600')
print("finish make fasttext 600 vector dataset")

print("start make fasttext 900 vector dataset")
fast900_model = FastText.load(fast900_model_dir)
vector_maker(fast900_model, word_vocab, all_vector_fast900, MS_label, embedding_size=900)
vector_maker(fast900_model, word_vocab, all_vector_fast900, train_label, embedding_size=900)
vector_maker(fast900_model, word_vocab, all_vector_fast900, pre_label, embedding_size=900)
vector_maker(fast900_model, word_vocab, all_vector_fast900, final1_label, embedding_size=900)
vector_maker(fast900_model, word_vocab, all_vector_fast900, final2_label, embedding_size=900)
make_vector_dataset(all_vector_fast900, embedding_file_name='fasttext900')
print("finish make fasttext 900 vector dataset")
'''

'''

#check_mean(train_label)
#check_mean(pre_label)
#check_mean(final1_label)
#check_mean(final2_label)


#New_all_dataMaker(train_label)
#New_all_dataMaker(pre_label)
#New_all_dataMaker(final1_label)
#New_all_dataMaker(final2_label)

#New_make_data_set()


#opcode_overlap_checker(train_label)
#opcode_overlap_checker(pre_label)
#opcode_overlap_checker(final1_label)
#opcode_overlap_checker(final2_label)

#opcode_preprocessor(train_label)
#opcode_preprocessor(pre_label)
#opcode_preprocessor(final1_label)
#opcode_preprocessor(final2_label)

#api_checker()

#api_preprocessor(train_label)
#api_preprocessor(pre_label)
#api_preprocessor(final1_label)
#api_preprocessor(final2_label)

# Dataset이 label##name##contents로 이루어진 txt파일을 만드는 함수
def New_all_dataMaker(label_file):
    all_data_element = open('./dataset/new_all_data_4.txt', 'a')
    csv_data = open(label_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_data)
    for line in csv_reader:
        try:
            # opcode 파일 열어서 opcode contents에 사용할 거 추출
            file = open('./dataset/preprocessing_opcode/%s.txt' % line[0], 'r')

            all_data_element.write(str(line[1]) + '##')
            # opcode 들을 list에 저장하여 all_data.txt에 넣을 contents 만들기
            i = 0    # 3600개 개수 세어줄 것
            k = 0    # 8개 개수 세어줄 것
            for opline in file:
                opline = opline.replace(' ', '')
                if k >= 8:
                    all_data_element.write(' ')
                    k = 0

                if opline.find('?') != -1 and opline.find('nop'):
                    print(opline)
                    pass

                if i >= 4794:
                    break

                else:
                    all_data_element.write(opline.strip())
                    i += 1
                    k += 1
            file.close()
            all_data_element.write(' ')

            try:
                # API 파일 열어서 API contents에 사용할 거 추출
                file = open('./dataset/preprocessing_api/%s.txt' % line[0], 'r')

                # api 들을 list에 저장하여 all_data.txt에 넣을 contents 만들기
                j = 0
                for APIline in file:
                    all_data_element.write(APIline.strip())
                    j += 1
                    if j >= 150:
                        break

                all_data_element.write('\n')
                file.close()
            except:
                all_data_element.write('\n')
                file.close()
                pass
        except:
            pass

    csv_data.close()

def opcode_preprocessor(label_file):
    csv_data = open(label_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_data)
    for line in csv_reader:
        try:
            # opcode 파일 열어서 opcode contents에 사용할 거 추출
            file = open('./dataset/opcode/%s.txt' % line[0].strip(), 'r')
            processing_file = open('./dataset/preprocessing_opcode/%s.txt' % line[0].strip(), 'w')
            # opcode 들을 list에 저장하여 all_data.txt에 넣을 contents 만들기
            i = 0
            compressor_list = []
            for opline in file:
                if i == 0 or compressor_list[0] == False:
                    before = opline
                    compressor_list.append(before + '\n')
                elif i >= 1 and before == opline:
                    compressor_list.append(opline + '\n')
                elif i >= 1 and before != opline:
                    processing_file.write(compressor_list)
                    del compressor_list
                    compressor_list = []

            file.close()
            processing_file.close()
        except:
            print("error")
            pass

    csv_data.close()

def opcode_overlap_checker(label_file):
    csv_data = open(label_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_data)
    check_file = open('./dataset/opcode_overlap_check.txt', 'a')
    for line in csv_reader:
        try:
            check_list = []
            # opcode 파일 열어서 opcode contents에 사용할 거 추출
            file = open('./dataset/opcode/%s.txt' % line[0].strip(), 'r')
            i = 0
            check_cnt = 0
            for opline in file:
                if i == 0:
                    before = opline
                elif i >= 1 and before == opline:
                    check_cnt += 1
                elif i >= 1 and before != opline:
                    check_file.write(line[0].strip() + '\n' + before.strip() + check_cnt)


            file.close()
        except:
            print("error")
            pass

    csv_data.close()
    
    
    

# ./data/train/ 경로에 train, valid, test set을 text파일로 생성
def New_make_data_set():
    tempo = []
    with open('./dataset/new_all_data_4.txt','r') as all_data:
        for line in all_data:
            if len(line) > 5:
                tempo.append(line)
    size = len(tempo)//100
    train = tempo[0:80*size]
    valid = tempo[80*size:90*size]
    test = tempo[90*size:100*size]
    save = open('./data/train/new_train_4.txt','w')
    for line in train:
        print(line,file=save)

    save = open('./data/train/new_valid_4.txt','w')
    for line in valid:
        print(line,file=save)

    save = open('./data/train/new_test_4.txt','w')
    for line in test:
        print(line,file=save)

def api_preprocessor(label_file):
    csv_data = open(label_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_data)
    for line in csv_reader:
        try:
            # API 파일 열어서 API contents에 사용할 거 추출
            file = open('./dataset/api/%s.txt' % line[0].strip(), 'r')
            processing_file = open('./dataset/preprocessing_api/%s.txt' % line[0].strip(), 'w')
            # opcode 들을 list에 저장하여 all_data.txt에 넣을 contents 만들기
            for APIline in file:
                if APIline.find('_') == -1 and APIline.find('?') == -1 and APIline.find('@') == -1 and APIline.find('none') == -1 and APIline.find('None') == -1:
                    processing_file.write(APIline.strip() + '\n')

                else:
                    pass

            file.close()
            processing_file.close()
        except:
            print("error")
            pass

    csv_data.close()

def api_checker():
    api_list = os.listdir('./dataset/api')
    preprocessing_list = os.listdir('./dataset/preprocessing_api')

    nonfile_list = list(set(api_list) - set(preprocessing_list))
    print(nonfile_list)

def check_mean(label_file):
    csv_data = open(label_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_data)
    op_cnt = 0
    api_cnt = 0
    wrong_cnt = 0
    for line in csv_reader:
        try:
            # opcode 파일 열어서 opcode contents에 사용할 거 추출
            op_file = open('./dataset/preprocessing_opcode/%s.txt' % line[0], 'r')
            for opline in op_file:
                op_cnt += 1

        except:
            #print(line[0] + " error!")
            pass

        try:
            # opcode 파일 열어서 opcode contents에 사용할 거 추출
            api_file = open('./dataset/preprocessing_api/%s.txt' % line[0], 'r')
            for apiline in api_file:
                api_cnt += 1

        except:
            #print(line[0] + " error!")
            pass
    print("op_cnt = " + str(op_cnt))
    print("API_cnt = " + str(api_cnt))
'''