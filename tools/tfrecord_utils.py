#coding:utf-8
import tensorflow as tf
import os
import cv2
from PIL import Image


def _int64_feature(value):
    """Wrapper for insert int64 feature into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for insert float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for insert bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _convert_to_example(image_example, image_buffer, colorspace=b'RGB', channels=3, image_format=b'JPEG'):
    """
    covert to tfrecord file
    :param image_example: dict, an image example
    :param image_buffer: string, JPEG encoding of RGB image
    :param colorspace:
    :param channels:
    :param image_format:
    :return:
    Example proto
    """
    # filename = str(image_example['filename'])

    # class label for the whole image
    class_label = image_example['label']
    # print(class_label)

    image_bboxes = image_example.get('bbox', {})
    xmin = image_bboxes.get('xmin', [])
    xmax = image_bboxes.get('xmax', [])
    ymin = image_bboxes.get('ymin', [])
    ymax = image_bboxes.get('ymax', [])
    # print(xmin)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/format': _bytes_feature(image_format),
        'image/encoded': _bytes_feature(image_buffer),
        'image/label': _int64_feature(class_label),
        'image/image_bbox/xmin': _float_feature(xmin),
        'image/image_bbox/ymin': _float_feature(ymin),
        'image/image_bbox/xmax': _float_feature(xmax),
        'image/image_bbox/ymax': _float_feature(ymax),
    }))
    return example
def _convert_to_example_simple(image_example, image_buffer):
    """
    covert to tfrecord file
    :param image_example: dict, an image example
    :param image_buffer: string, JPEG encoding of RGB image
    :param colorspace:
    :param channels:
    :param image_format:
    :return:
    Example proto
    """
    # filename = str(image_example['filename'])

    # class label for the whole image
    class_label = image_example['label']
    bbox = image_example['bbox']
    roi = [bbox['xmin'],bbox['ymin'],bbox['xmax'],bbox['ymax']]
    landmark = [bbox['jaw_0_x'],bbox['jaw_0_y'],bbox['jaw_1_x'],bbox['jaw_1_y'],bbox['jaw_2_x'],bbox['jaw_2_y'],bbox['jaw_3_x'],bbox['jaw_3_y'],bbox['jaw_4_x'],bbox['jaw_4_y'],bbox['jaw_5_x'],bbox['jaw_5_y'],bbox['jaw_6_x'],bbox['jaw_6_y'],bbox['jaw_7_x'],bbox['jaw_7_y'],bbox['jaw_8_x'],bbox['jaw_8_y'],bbox['jaw_9_x'],bbox['jaw_9_y'],bbox['jaw_10_x'],bbox['jaw_10_y'],bbox['jaw_11_x'],bbox['jaw_11_y'],bbox['jaw_12_x'],bbox['jaw_12_y'],bbox['jaw_13_x'],bbox['jaw_13_y'],bbox['jaw_14_x'],bbox['jaw_14_y'],bbox['jaw_15_x'],bbox['jaw_15_y'],bbox['jaw_16_x'],bbox['jaw_16_y'],bbox['left_eyebrow_0_x'],bbox['left_eyebrow_0_y'],bbox['left_eyebrow_1_x'],bbox['left_eyebrow_1_y'],bbox['left_eyebrow_2_x'],bbox['left_eyebrow_2_y'],bbox['left_eyebrow_3_x'],bbox['left_eyebrow_3_y'],bbox['left_eyebrow_4_x'],bbox['left_eyebrow_4_y'],bbox['right_eyebrow_0_x'],bbox['right_eyebrow_0_y'],bbox['right_eyebrow_1_x'],bbox['right_eyebrow_1_y'],bbox['right_eyebrow_2_x'],bbox['right_eyebrow_2_y'],bbox['right_eyebrow_3_x'],bbox['right_eyebrow_3_y'],bbox['right_eyebrow_4_x'],bbox['right_eyebrow_4_y'],bbox['noze_0_x'],bbox['noze_0_y'],bbox['noze_1_x'],bbox['noze_1_y'],bbox['noze_2_x'],bbox['noze_2_y'],bbox['noze_3_x'],bbox['noze_3_y'],bbox['noze_4_x'],bbox['noze_4_y'],bbox['noze_5_x'],bbox['noze_5_y'],bbox['noze_6_x'],bbox['noze_6_y'],bbox['noze_7_x'],bbox['noze_7_y'],bbox['noze_8_x'],bbox['noze_8_y'],bbox['left_eye_0_x'],bbox['left_eye_0_y'],bbox['left_eye_1_x'],bbox['left_eye_1_y'],bbox['left_eye_2_x'],bbox['left_eye_2_y'],bbox['left_eye_3_x'],bbox['left_eye_3_y'],bbox['left_eye_4_x'],bbox['left_eye_4_y'],bbox['left_eye_5_x'],bbox['left_eye_5_y'],bbox['right_eye_0_x'],bbox['right_eye_0_y'],bbox['right_eye_1_x'],bbox['right_eye_1_y'],bbox['right_eye_2_x'],bbox['right_eye_2_y'],bbox['right_eye_3_x'],bbox['right_eye_3_y'],bbox['right_eye_4_x'],bbox['right_eye_4_y'],bbox['right_eye_5_x'],bbox['right_eye_5_y'],bbox['mouth_0_x'],bbox['mouth_0_y'],bbox['mouth_1_x'],bbox['mouth_1_y'],bbox['mouth_2_x'],bbox['mouth_2_y'],bbox['mouth_3_x'],bbox['mouth_3_y'],bbox['mouth_4_x'],bbox['mouth_4_y'],bbox['mouth_5_x'],bbox['mouth_5_y'],bbox['mouth_6_x'],bbox['mouth_6_y'],bbox['mouth_7_x'],bbox['mouth_7_y'],bbox['mouth_8_x'],bbox['mouth_8_y'],bbox['mouth_9_x'],bbox['mouth_9_y'],bbox['mouth_10_x'],bbox['mouth_10_y'],bbox['mouth_11_x'],bbox['mouth_11_y'],bbox['mouth_12_x'],bbox['mouth_12_y'],bbox['mouth_13_x'],bbox['mouth_13_y'],bbox['mouth_14_x'],bbox['mouth_14_y'],bbox['mouth_15_x'],bbox['mouth_15_y'],bbox['mouth_16_x'],bbox['mouth_16_y'],bbox['mouth_17_x'],bbox['mouth_17_y'],bbox['mouth_18_x'],bbox['mouth_18_y'],bbox['mouth_19_x'],bbox['mouth_19_y']]
                
      
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_buffer),
        'image/label': _int64_feature(class_label),
        'image/roi': _float_feature(roi),
        'image/landmark': _float_feature(landmark)
    }))
    return example



class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""
    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        # Convert the image data from png to jpg
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        # Decode the image data as a jpeg image
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3, "JPEG needs to have height x width x channels"
        assert image.shape[2] == 3, "JPEG needs to have 3 channels (RGB)"
        return image


def _is_png(filename):
    """Determine if a file contains a PNG format image.
    Args:
      filename: string, path of the image file.
    Returns:
      boolean indicating if the image is a PNG.
    """
    _, file_extension = os.path.splitext(filename)
    return file_extension.lower() == '.png'


def _process_image(filename, coder):
    """Process a single image file.
    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    # Note: tf bug 使用‘r‘会出错，无法解码，只能以2进制形式读取
    # image_data = tf.gfile.FastGFile(filename, 'r').read()
    # image_raw_data = tf.gfile.FastGFile(filename, 'rb').read()
    # img_data_jpg = tf.image.decode_jpeg(image_raw_data)
    # img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)
    # resized_image = tf.image.resize_images(img_data_jpg, [25, 25])
    # image_data = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
    # image = Image.open(filename)  # 图片的类型必须为array
    filename = filename + '.jpg'
    print filename
    image = cv2.imread(filename)
    # image.show()
    # image_data = image.tobytes()
    image_data = image.tostring()

    # Clean the dirty data.
    if _is_png(filename):
        print(filename, 'to convert jpeg')
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    # image = coder.decode_jpeg(image_data)
    # print(image.shape)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width
def _process_image_withoutcoder(filename):
    image = cv2.imread(filename)
    image_data = image.tostring()
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width



