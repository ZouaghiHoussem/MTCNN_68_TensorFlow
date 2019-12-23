# coding: utf-8
import sys, os
rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)
import numpy as np
import argparse

from tools.tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple
import tensorflow as tf


def __iter_all_data(net, iterType):
    saveFolder = os.path.join(rootPath, "tmp/data/%s/"%(net))
    if net not in ['pnet', 'rnet', 'onet']:
        raise Exception("The net type error!")
    if not os.path.isfile(os.path.join(saveFolder, 'pos.txt')):
        raise Exception("Please gen pos.txt in first!")
    if not os.path.isfile(os.path.join(saveFolder, 'landmark.txt')):
        raise Exception("Please gen landmark.txt in first!")
    if iterType == 'all':
        with open(os.path.join(saveFolder, 'pos.txt'), 'r') as f:
            pos = f.readlines()
        with open(os.path.join(saveFolder, 'neg.txt'), 'r') as f:
            neg = f.readlines()
        with open(os.path.join(saveFolder, 'part.txt'), 'r') as f:
            part = f.readlines()
        # keep sample ratio [neg, pos, part] = [3, 1, 1]
        base_num = min([len(neg), len(pos), len(part)])
        if len(neg) > base_num * 3:
            neg_keep = np.random.choice(len(neg), size=base_num * 3, replace=False)
        else:
            neg_keep = np.random.choice(len(neg), size=len(neg), replace=False)
        pos_keep = np.random.choice(len(pos), size=base_num, replace=False)
        part_keep = np.random.choice(len(part), size=base_num, replace=False)
        for i in pos_keep:
            yield pos[i]
        for i in neg_keep:
            yield neg[i]
        for i in part_keep:
            yield part[i]
        for item in open(os.path.join(saveFolder, 'landmark.txt'), 'r'):
            yield item 
    elif iterType in ['pos', 'neg', 'part', 'landmark']:
        for line in open(os.path.join(saveFolder, '%s.txt'%(iterType))):
            yield line
    else:
        raise Exception("Unsupport iter type.")

def __get_dataset(net, iterType):
    dataset = []
    for line in __iter_all_data(net, iterType):
        info = line.strip().split(' ')
        data_example = dict()
        bbox = dict()
        data_example['filename'] = info[0]
        data_example['label'] = int(info[1])
        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        bbox['jaw_0_x']= 0
        bbox['jaw_0_y']= 0
        bbox['jaw_1_x']= 0
        bbox['jaw_1_y']= 0
        bbox['jaw_2_x']= 0
        bbox['jaw_2_y']= 0
        bbox['jaw_3_x']= 0
        bbox['jaw_3_y']= 0
        bbox['jaw_4_x']= 0
        bbox['jaw_4_y']= 0
        bbox['jaw_5_x']= 0
        bbox['jaw_5_y']= 0
        bbox['jaw_6_x']= 0
        bbox['jaw_6_y']= 0
        bbox['jaw_7_x']= 0
        bbox['jaw_7_y']= 0
        bbox['jaw_8_x']= 0
        bbox['jaw_8_y']= 0
        bbox['jaw_9_x']= 0
        bbox['jaw_9_y']= 0
        bbox['jaw_10_x']= 0
        bbox['jaw_10_y']= 0
        bbox['jaw_11_x']= 0
        bbox['jaw_11_y']= 0
        bbox['jaw_12_x']= 0
        bbox['jaw_12_y']= 0
        bbox['jaw_13_x']= 0
        bbox['jaw_13_y']= 0
        bbox['jaw_14_x']= 0
        bbox['jaw_14_y']= 0
        bbox['jaw_15_x']= 0
        bbox['jaw_15_y']= 0
        bbox['jaw_16_x']= 0
        bbox['jaw_16_y']= 0        
        bbox['left_eyebrow_0_x']= 0
        bbox['left_eyebrow_0_y']= 0
        bbox['left_eyebrow_1_x']= 0
        bbox['left_eyebrow_1_y']= 0
        bbox['left_eyebrow_2_x']= 0
        bbox['left_eyebrow_2_y']= 0
        bbox['left_eyebrow_3_x']= 0
        bbox['left_eyebrow_3_y']= 0
        bbox['left_eyebrow_4_x']= 0
        bbox['left_eyebrow_4_y']= 0
        bbox['right_eyebrow_0_x']=0
        bbox['right_eyebrow_0_y']=0
        bbox['right_eyebrow_1_x']=0
        bbox['right_eyebrow_1_y']=0
        bbox['right_eyebrow_2_x']=0
        bbox['right_eyebrow_2_y']=0
        bbox['right_eyebrow_3_x']=0
        bbox['right_eyebrow_3_y']=0
        bbox['right_eyebrow_4_x']=0
        bbox['right_eyebrow_4_y']=0
        bbox['noze_0_x']=0
        bbox['noze_0_y']=0
        bbox['noze_1_x']=0
        bbox['noze_1_y']=0
        bbox['noze_2_x']=0
        bbox['noze_2_y']=0
        bbox['noze_3_x']=0
        bbox['noze_3_y']=0
        bbox['noze_4_x']=0
        bbox['noze_4_y']=0
        bbox['noze_5_x']=0
        bbox['noze_5_y']=0
        bbox['noze_6_x']=0
        bbox['noze_6_y']=0
        bbox['noze_7_x']=0
        bbox['noze_7_y']=0
        bbox['noze_8_x']=0
        bbox['noze_8_y']=0
        bbox['left_eye_0_x']=0
        bbox['left_eye_0_y']=0
        bbox['left_eye_1_x']=0
        bbox['left_eye_1_y']=0
        bbox['left_eye_2_x']=0
        bbox['left_eye_2_y']=0
        bbox['left_eye_3_x']=0
        bbox['left_eye_3_y']=0
        bbox['left_eye_4_x']=0
        bbox['left_eye_4_y']=0
        bbox['left_eye_5_x']=0
        bbox['left_eye_5_y']=0
        bbox['right_eye_0_x']=0
        bbox['right_eye_0_y']=0
        bbox['right_eye_1_x']=0
        bbox['right_eye_1_y']=0
        bbox['right_eye_2_x']=0
        bbox['right_eye_2_y']=0
        bbox['right_eye_3_x']=0
        bbox['right_eye_3_y']=0
        bbox['right_eye_4_x']=0
        bbox['right_eye_4_y']=0
        bbox['right_eye_5_x']=0
        bbox['right_eye_5_y']=0
        bbox['mouth_0_x']=0
        bbox['mouth_0_y']=0
        bbox['mouth_1_x']=0
        bbox['mouth_1_y']=0
        bbox['mouth_2_x']=0
        bbox['mouth_2_y']=0
        bbox['mouth_3_x']=0
        bbox['mouth_3_y']=0
        bbox['mouth_4_x']=0
        bbox['mouth_4_y']=0
        bbox['mouth_5_x']=0
        bbox['mouth_5_y']=0
        bbox['mouth_6_x']=0
        bbox['mouth_6_y']=0
        bbox['mouth_7_x']=0
        bbox['mouth_7_y']=0
        bbox['mouth_8_x']=0
        bbox['mouth_8_y']=0
        bbox['mouth_9_x']=0
        bbox['mouth_9_y']=0
        bbox['mouth_10_x']=0
        bbox['mouth_10_y']=0
        bbox['mouth_11_x']=0
        bbox['mouth_11_y']=0
        bbox['mouth_12_x']=0
        bbox['mouth_12_y']=0
        bbox['mouth_13_x']=0
        bbox['mouth_13_y']=0
        bbox['mouth_14_x']=0
        bbox['mouth_14_y']=0
        bbox['mouth_15_x']=0
        bbox['mouth_15_y']=0
        bbox['mouth_16_x']=0
        bbox['mouth_16_y']=0
        bbox['mouth_17_x']=0
        bbox['mouth_17_y']=0
        bbox['mouth_18_x']=0
        bbox['mouth_18_y']=0
        bbox['mouth_19_x']=0
        bbox['mouth_19_y']=0

        
        if len(info) == 6:
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        if len(info) == 138:
            #bbox['jaw'] = info[2:18]#[0:16]#
            #bbox['left_eyebrow'] = info[19:23]#[17:21]#
            #bbox['right_eyebrow'] = info[24:28]#[22:26]#
            #bbox['nose'] = info[29:37]#[27:35]#
            #bbox['left_eye'] = info[38:43]#[36:41]#
            #bbox['right_eye'] = info[44:49]#[42:47]#
            #bbox['mouth'] = info[50:69]#[48:67]#


            bbox['jaw_0_x']=float(info[2])
            bbox['jaw_0_y']=float(info[3])
            bbox['jaw_1_x']=float(info[4])
            bbox['jaw_1_y']=float(info[5])
            bbox['jaw_2_x']=float(info[6])
            bbox['jaw_2_y']=float(info[7])
            bbox['jaw_3_x']=float(info[8])
            bbox['jaw_3_y']=float(info[9])
            bbox['jaw_4_x']=float(info[10])
            bbox['jaw_4_y']=float(info[11])
            bbox['jaw_5_x']=float(info[12])
            bbox['jaw_5_y']=float(info[13])
            bbox['jaw_6_x']=float(info[14])
            bbox['jaw_6_y']=float(info[15])
            bbox['jaw_7_x']=float(info[16])
            bbox['jaw_7_y']=float(info[17])
            bbox['jaw_8_x']=float(info[18])
            bbox['jaw_8_y']=float(info[19])
            bbox['jaw_9_x']=float(info[20])
            bbox['jaw_9_y']=float(info[21])
            bbox['jaw_10_x']=float(info[22])
            bbox['jaw_10_y']=float(info[23])
            bbox['jaw_11_x']=float(info[24])
            bbox['jaw_11_y']=float(info[25])
            bbox['jaw_12_x']=float(info[26])
            bbox['jaw_12_y']=float(info[27])
            bbox['jaw_13_x']=float(info[28])
            bbox['jaw_13_y']=float(info[29])
            bbox['jaw_14_x']=float(info[30])
            bbox['jaw_14_y']=float(info[31])
            bbox['jaw_15_x']=float(info[32])
            bbox['jaw_15_y']=float(info[33])
            bbox['jaw_16_x']=float(info[34])
            bbox['jaw_16_y']=float(info[35])           
            bbox['left_eyebrow_0_x']=float(info[36])
            bbox['left_eyebrow_0_y']=float(info[37])
            bbox['left_eyebrow_1_x']=float(info[38])
            bbox['left_eyebrow_1_y']=float(info[39])
            bbox['left_eyebrow_2_x']=float(info[40])
            bbox['left_eyebrow_2_y']=float(info[41])
            bbox['left_eyebrow_3_x']=float(info[42])
            bbox['left_eyebrow_3_y']=float(info[43])
            bbox['left_eyebrow_4_x']=float(info[44])
            bbox['left_eyebrow_4_y']=float(info[45])
            bbox['right_eyebrow_0_x']=float(info[46])
            bbox['right_eyebrow_0_y']=float(info[47])
            bbox['right_eyebrow_1_x']=float(info[48])
            bbox['right_eyebrow_1_y']=float(info[49])
            bbox['right_eyebrow_2_x']=float(info[50])
            bbox['right_eyebrow_2_y']=float(info[51])
            bbox['right_eyebrow_3_x']=float(info[52])
            bbox['right_eyebrow_3_y']=float(info[53])
            bbox['right_eyebrow_4_x']=float(info[54])
            bbox['right_eyebrow_4_y']=float(info[55])
            bbox['noze_0_x']=float(info[56])
            bbox['noze_0_y']=float(info[57])
            bbox['noze_1_x']=float(info[58])
            bbox['noze_1_y']=float(info[59])
            bbox['noze_2_x']=float(info[60])
            bbox['noze_2_y']=float(info[61])
            bbox['noze_3_x']=float(info[62])
            bbox['noze_3_y']=float(info[63])
            bbox['noze_4_x']=float(info[64])
            bbox['noze_4_y']=float(info[65])
            bbox['noze_5_x']=float(info[66])
            bbox['noze_5_y']=float(info[67])
            bbox['noze_6_x']=float(info[68])
            bbox['noze_6_y']=float(info[69])
            bbox['noze_7_x']=float(info[70])
            bbox['noze_7_y']=float(info[71])
            bbox['noze_8_x']=float(info[72])
            bbox['noze_8_y']=float(info[73])
            bbox['left_eye_0_x']=float(info[74])
            bbox['left_eye_0_y']=float(info[75])
            bbox['left_eye_1_x']=float(info[76])
            bbox['left_eye_1_y']=float(info[77])
            bbox['left_eye_2_x']=float(info[78])
            bbox['left_eye_2_y']=float(info[79])
            bbox['left_eye_3_x']=float(info[80])
            bbox['left_eye_3_y']=float(info[81])
            bbox['left_eye_4_x']=float(info[82])
            bbox['left_eye_4_y']=float(info[83])
            bbox['left_eye_5_x']=float(info[84])
            bbox['left_eye_5_y']=float(info[85])
            bbox['right_eye_0_x']=float(info[86])
            bbox['right_eye_0_y']=float(info[87])
            bbox['right_eye_1_x']=float(info[88])
            bbox['right_eye_1_y']=float(info[89])
            bbox['right_eye_2_x']=float(info[90])
            bbox['right_eye_2_y']=float(info[91])
            bbox['right_eye_3_x']=float(info[92])
            bbox['right_eye_3_y']=float(info[93])
            bbox['right_eye_4_x']=float(info[94])
            bbox['right_eye_4_y']=float(info[95])
            bbox['right_eye_5_x']=float(info[96])
            bbox['right_eye_5_y']=float(info[97])
            bbox['mouth_0_x']=float(info[98])
            bbox['mouth_0_y']=float(info[99])
            bbox['mouth_1_x']=float(info[100])
            bbox['mouth_1_y']=float(info[101])
            bbox['mouth_2_x']=float(info[102])
            bbox['mouth_2_y']=float(info[103])
            bbox['mouth_3_x']=float(info[104])
            bbox['mouth_3_y']=float(info[105])
            bbox['mouth_4_x']=float(info[106])
            bbox['mouth_4_y']=float(info[107])
            bbox['mouth_5_x']=float(info[108])
            bbox['mouth_5_y']=float(info[109])
            bbox['mouth_6_x']=float(info[110])
            bbox['mouth_6_y']=float(info[111])
            bbox['mouth_7_x']=float(info[112])
            bbox['mouth_7_y']=float(info[113])
            bbox['mouth_8_x']=float(info[114])
            bbox['mouth_8_y']=float(info[115])
            bbox['mouth_9_x']=float(info[116])
            bbox['mouth_9_y']=float(info[117])
            bbox['mouth_10_x']=float(info[118])
            bbox['mouth_10_y']=float(info[119])
            bbox['mouth_11_x']=float(info[120])
            bbox['mouth_11_y']=float(info[121])
            bbox['mouth_12_x']=float(info[122])
            bbox['mouth_12_y']=float(info[123])
            bbox['mouth_13_x']=float(info[124])
            bbox['mouth_13_y']=float(info[125])
            bbox['mouth_14_x']=float(info[126])
            bbox['mouth_14_y']=float(info[127])
            bbox['mouth_15_x']=float(info[128])
            bbox['mouth_15_y']=float(info[129])
            bbox['mouth_16_x']=float(info[130])
            bbox['mouth_16_y']=float(info[131])
            bbox['mouth_17_x']=float(info[132])
            bbox['mouth_17_y']=float(info[133])
            bbox['mouth_18_x']=float(info[134])
            bbox['mouth_18_y']=float(info[135])
            bbox['mouth_19_x']=float(info[136])
            bbox['mouth_19_y']=float(info[137])


        data_example['bbox'] = bbox
        dataset.append(data_example)
    return dataset

def __add_to_tfrecord(filename, image_example, tfrecord_writer):
    """
    Loads data from image and annotations files and add them to a TFRecord.
    """
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())

def gen_tfrecords(net, shuffling=False):
    """
    Runs the conversion operation.
    """
    print(">>>>>> Start tfrecord create...Stage: %s"%(net))
    def _gen(tfFileName, net, iterType, shuffling):
        if tf.gfile.Exists(tfFileName):
            tf.gfile.Remove(tfFileName)
        # GET Dataset, and shuffling.
        dataset = __get_dataset(net=net, iterType=iterType)
        if shuffling:
            np.random.shuffle(dataset)
        # Process dataset files.
        # write the data to tfrecord
        with tf.python_io.TFRecordWriter(tfFileName) as tfrecord_writer:
            for i, image_example in enumerate(dataset):
                if i % 100 == 0:
                    sys.stdout.write('\rConverting[%s]: %d/%d' % (net, i + 1, len(dataset)))
                    sys.stdout.flush()
                filename = image_example['filename']
                __add_to_tfrecord(filename, image_example, tfrecord_writer)
        tfrecord_writer.close()
        print('\n')
    saveFolder = os.path.join(rootPath, "tmp/data/%s/"%(net))
    #tfrecord name 
    if net == 'pnet':
        tfFileName = os.path.join(saveFolder, "all.tfrecord")
        _gen(tfFileName, net, 'all', shuffling)
    elif net in ['rnet', 'onet']:
        for n in ['pos', 'neg', 'part', 'landmark']:
            tfFileName = os.path.join(saveFolder, "%s.tfrecord"%(n))
            _gen(tfFileName, net, n, shuffling)
    # Finally, write the labels file:
    print('\nFinished converting the MTCNN dataset!')
    print('All tf record was saved in %s'%(saveFolder))

def parse_args():
    parser = argparse.ArgumentParser(description='Create hard bbox sample...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', dest='stage', help='working stage, can be pnet, rnet, onet',
                        default='unknow', type=str)
    parser.add_argument('--gpus', dest='gpus', help='specify gpu to run. eg: --gpus=0,1',
                        default='0', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    stage = args.stage
    if stage not in ['pnet', 'rnet', 'onet']:
        raise Exception("Please specify stage by --stage=pnet or rnet or onet")
    # set GPU
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    gen_tfrecords(stage, True)

