import os
import sys
sys.path.append("..")
import time
import pickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
from io import StringIO as StringIO
import base64
from io import BytesIO
import urllib
import exifutil
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from keras.preprocessing import image
#from skimage import io, transform
#from network.hiarBayesGoogLenet import hiarBayesGoogLeNet
from inception.build_models import *
#from sklearn.metrics import accuracy_score


REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOAD_FOLDER = REPO_DIRNAME + '/data/tmp/pa_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# Obtain the flask app object
app = flask.Flask(__name__)

has_metrics=False
metrics=None
has_result=False
result=None
imagesrc=None
has_attributes=False
attributes=None
imageurl = None
image_show = None


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    global imageurl
    imageurl = flask.request.args.get('imageurl', '')
    try:
        """
        string_buffer = StringIO(
            urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)
        """
        image = io.imread(imageurl)
        image = transform.resize(image, (160, 75))

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_metrics=False, metrics=None, has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    global result
    result = app.clf.classify_image(image)
    global has_result
    has_result==True
    return flask.render_template(
        'index.html', has_metrics=has_metrics, metrics=metrics, 
        has_attributes=has_attributes, attributes=attributes,
        has_result=has_result, result=result, 
        imagesrc=imageurl)


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        global image_show
        image_show = exifutil.open_oriented_im(filename)
        img = image.load_img(filename, target_size=(224, 224, 3))
        image_resize = image.img_to_array(img)
        print(type(image_resize))
        print(np.shape(image_resize))
        

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    global result 
    result = app.clf.classify_image(image_resize)
    global has_result
    has_result=True
    #imagesrc=embed_image_html(image_show)
    return flask.render_template(
        'index.html', has_metrics=has_metrics, metrics=metrics, 
        has_attributes=has_attributes, attributes=attributes,
        has_result=has_result, result=result,
        imagesrc=embed_image_html(image_show)
    )

"""
@app.route('/classify_metrics', methods=['GET'])
def classify_metrics():
    logging.info('Loading the PETA dataset...')
    
    global metrics
    metrics = app.clf.classify_metrics()
    global has_metrics
    has_metrics=True
    if imageurl is not None:
        image = imageurl
    elif image_show is not None:
        image = embed_image_html(image_show)
    else:
        image = ""
    #imagesrc=embed_image_html(image_show)
    return flask.render_template(
        'index.html', has_metrics=has_metrics, metrics=metrics, 
        has_attributes=has_attributes, attributes=attributes,
        has_result=has_result, result=result,
        imagesrc=image
    )
    #"""


"""
@app.route('/attributes_metrics', methods=['GET'])
def attributes_metrics():
    logging.info('Loading the PETA dataset...')
    
    global atributes
    attributes = app.clf.attributes_metrics()
    global has_attributes
    has_attributes=True
    if imageurl is not None:
        image = imageurl
    elif image_show is not None:
        image = embed_image_html(image_show)
    else:
        image = ""
    return flask.render_template(
        'index.html', has_metrics=has_metrics, metrics=metrics, 
        has_attributes=has_attributes, attributes=attributes,
        has_result=has_result, result=result,
        imagesrc=image
    )
    #"""


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    #image_pil = image_pil.resize((160, 75))
    #string_buf = StringIO()
    #image_pil.save(string_buf, format='png')
    #data = string_buf.getvalue().encode('base64').replace('\n', '')
    string_buf = BytesIO()
    image_pil.save(string_buf, format="png")
    data = base64.b64encode(string_buf.getvalue())
    print(type(data))
    #print(str(data))
    #return 'data:image/png;base64,' + data
    return "data:image/png;base64," + bytes.decode(data)
    #return str(data)


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


class ImagenetClassifier(object):
    """
    default_args = {
        'model_def_file': (
            '{}/models/bvlc_reference_caffenet/deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'.format(REPO_DIRNAME)),
        'mean_file': (
            '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
        'class_labels_file': (
            '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
        'bet_file': (
            '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
    }
    for key, val in default_args.items():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255.
    """
    default_args = {
        'model_def_file_RAP': (
            '{}/models/RAP/HRPInception/v2_epoch053_valloss0.859308.hdf5'.format(REPO_DIRNAME)),
        'model_def_file_PETA': (
            '{}/models/PETA/HRPInception/v2_epoch494_valloss0.986102.hdf5'.format(REPO_DIRNAME)),
        'model_def_file_PETA_g': (
            '{}/models/PETA/Inception/v1_epoch054_valloss0.985669.hdf5'.format(REPO_DIRNAME)),
        'attributes_PETA_g': (
            '{}/../spatial_attribute/results/PETA/Inception/_attributes_list.txt'.format(REPO_DIRNAME)),
        'attributes_PETA': (
            '{}/results/PETA/HRPInception/thr3_attributes_list.txt'.format(REPO_DIRNAME)),
        'attributes_RAP': (
            '{}/results/RAP/HRPInception/thr3_attributes_list.txt'.format(REPO_DIRNAME)),
    }
    for key, val in default_args.items():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_height'] = 224
    default_args['image_width'] = 224
    default_args['raw_scale'] = 255.

    def __init__(self, model_def_file_RAP, attributes_RAP, model_def_file_PETA, model_def_file_PETA_g, attributes_PETA_g, attributes_PETA, image_height, image_width, raw_scale, gpu_mode):
        logging.info('Loading net and associated files...')
        if not gpu_mode:
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
        low_peta = [27,32,50,56]
        hs_peta = [0,8,20,21,25,28,36,37,44,54]
        ub_peta = [23,30,39,46,51,55,58,59,60]
        lb_peta = [22,24,29,45,47,53,57]
        sh_peta = [9,26,42,43,48,49]
        at_peta = [6,7,11,12,13,17,33,35,38,41,52]
        high_whole = [1,2,3,4,5,16,34]
        high_ub = [15,19,40]
        high_lb = [10,14,18,31]
        self.parts_peta = [[len(low_peta)], [len(hs_peta), len(ub_peta), len(lb_peta), len(sh_peta), len(at_peta)], [len(high_whole), len(high_ub), len(high_lb)]]
        self.idx_indices_peta = list(np.hstack((low_peta, hs_peta, ub_peta, lb_peta, sh_peta, at_peta, high_whole, high_ub, high_lb)))
        """
        seg_graph_peta = tf.Graph()
        sess_peta = tf.Session(graph=seg_graph_peta)
        K.set_session(sess_peta)
        with seg_graph_peta.as_default():
            net_peta = build_hrp_inception(self.parts_peta, version="v2", width=image_width, height=image_height)
            net_peta.load_weights(model_def_file_PETA)
            #"""
        
        self.net_peta = build_hrp_inception(self.parts_peta, version="v2", width=image_width, height=image_height)
        self.net_peta.load_weights(model_def_file_PETA)
        f = open(attributes_PETA)
        self.attributes_peta = f.readlines()
        for i in range(0, len(self.attributes_peta)):
            self.attributes_peta[i] = self.attributes_peta[i].strip()
            
        self.net_peta_g = build_orig_inception(len(self.idx_indices_peta), width=image_width, height=image_height)
        self.net_peta_g.load_weights(model_def_file_PETA_g)
        f = open(attributes_PETA_g)
        self.attributes_peta_g = f.readlines()
        for i in range(0, len(self.attributes_peta_g)):
            self.attributes_peta_g[i] = self.attributes_peta_g[i].strip()
        
        low = [11]
        hs = [9,10,12,13,14]
        ub = [15,16,17,18,19,20,21,22,23]
        lb = [24,25,26,27,28,29]
        sh = [30,31,32,33,34]
        at = [35,36,37,38,39,40,41,42]
        high = [0,1,2,3,4,5,6,7,8,43,44,45,46,47,48,49,50]
        self.parts_rap = [[len(low)], [len(hs), len(ub), len(lb), len(sh), len(at)], [len(high)]]
        self.idx_indices_rap = list(np.hstack((low, hs, ub, lb, sh, at, high)))
        """
        seg_graph_rap = tf.Graph()
        sess_rap = tf.Session(graph=seg_graph_rap)
        K.set_session(sess_rap)
        with seg_graph_rap.as_default():
            net_rap = build_hrp_inception(self.parts_rap, version="v2", width=image_width, height=image_height)
            net_rap.load_weights(model_def_file_RAP)
            #"""
        """
        self.net_rap = build_hrp_inception(self.parts_rap, version="v2", width=image_width, height=image_height)
        self.net_rap.load_weights(model_def_file_RAP)
        #"""
        f = open(attributes_RAP)
        self.attributes_rap = f.readlines()
        for i in range(0, len(self.attributes_rap)):
            self.attributes_rap[i] = self.attributes_rap[i].strip()
            

    def classify_image(self, image):
        try:
            image_reshape = image[np.newaxis, :, :, :]
            print(image_reshape.shape)
            starttime = time.time()
            """
            K.clear_session()
            net_peta = build_hrp_inception(self.parts_peta, version="v2", width=self.default_args['image_width'], height=self.default_args['image_height'])
            net_peta.load_weights(self.default_args['model_def_file_PETA'])
            #"""
            
            scores = self.net_peta_g.predict(image_reshape).flatten()

            indices = []
            for i in range(len(scores)):
                if scores[i] >= 0.5:
                    indices.append(i)
            predictions = []
            for i in indices:
                predictions.append(self.attributes_peta_g[i])
                #predictions.append(self.attributes_rap[i])

            # In addition to the prediction text, we will also produce
            # the length for the progress bar visualization.
            meta_g = [
                (p, '%.5f' % scores[i])
                for i, p in zip(indices, predictions)
            ]
            logging.info('result: %s', str(meta_g))
            
            scores = self.net_peta.predict(image_reshape).flatten()
            """
            K.clear_session()
            net_rap = build_hrp_inception(self.parts_rap, version="v2", width=self.default_args['image_width'], height=self.default_args['image_height'])
            net_rap.load_weights(self.default_args['model_def_file_RAP'])
            #"""
            #scores = self.net_rap.predict(image_reshape).flatten()
            endtime = time.time()

            indices = []
            for i in range(len(scores)):
                if scores[i] >= 0.5:
                    indices.append(i)
            predictions = []
            for i in indices:
                predictions.append(self.attributes_peta[i])
                #predictions.append(self.attributes_rap[i])

            # In addition to the prediction text, we will also produce
            # the length for the progress bar visualization.
            meta = [
                (p, '%.5f' % scores[i])
                for i, p in zip(indices, predictions)
            ]
            logging.info('result: %s', str(meta))

            # Compute expected information gain
            """
            expected_infogain = np.dot(
                self.bet['probmat'], scores[self.bet['idmapping']])
            expected_infogain *= self.bet['infogain']
            """

            # sort the scores
            """
            infogain_sort = expected_infogain.argsort()[::-1]
            bet_result = [(self.bet['words'][v], '%.5f' % expected_infogain[v])
                          for v in infogain_sort[:5]]
            logging.info('bet result: %s', str(bet_result))
            """

            return (True, meta_g, meta, '%.3f' % (endtime - starttime))

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')
    
    """
    def classify_metrics(self):
        filename = REPO_DIRNAME+"/results/PETA.csv"
        data = np.array(pd.read_csv(filename))[:, 1:]
        length = len(data)
        data_x = np.zeros((length, 160, 75, 3))
        data_y = np.zeros((length, 61))
        for i in range(11400, length):
            #img = image.load_img(path + m)
            img = image.load_img(data[i, 0], target_size=(160, 75, 3))
            data_x[i] = image.img_to_array(img)
            data_y[i] = np.array(data[i, 1:1+61], dtype="float32")
        data_y = data_y[:, list(np.hstack((self.low_level, self.mid_level, self.high_level)))]
        X_test = data_x[11400:]
        y_test = data_y[11400:]
        print("The shape of the X_test is: ", X_test.shape)
        print("The shape of the y_test is: ", y_test.shape)
        
        starttime = time.time()
        predictions = self.net.predict(X_test)
        np.save("../../results/predictions/app.npy", predictions)
        predictions = np.array(predictions >= 0.5, dtype="float64")
        self.predictions = predictions
        self.label = y_test

        def mA(y_pred, y_true):
            M = len(y_pred)
            L = len(y_pred[0])
            res = 0
            for i in range(L):
                P = sum(y_true[:, i])
                N = M - P
                TP = sum(y_pred[:, i]*y_true[:, i])
                TN = list(y_pred[:, i]+y_true[:, i] == 0).count(True)
                #print(P,',', N,',', TP,',', TN)
                if P != 0:
                    res += TP/P + TN/N
                else:
                    res += TN/N
            return res / (2*L)

        def acc(y_pred, y_true):
            M = len(y_pred)
            M_ = 0
            res = 0
            for i in range(M):
                #print(np.shape(y_pred[i]*y_true[i]))
                if sum(y_pred[i])+sum(y_true[i])-sum(y_pred[i]*y_true[i]) != 0:
                    res += sum(y_pred[i]*y_true[i]) / (sum(y_pred[i])+sum(y_true[i])-sum(y_pred[i]*y_true[i]))
                    M_ += 1
            return res / M_

        def prec(y_pred, y_true):
            M = len(y_pred)
            M_ = 0
            res = 0
            for i in range(M):
                if sum(y_pred[i]) != 0:
                    res += sum(y_pred[i]*y_true[i]) / sum(y_pred[i])
                    M_ += 1
            if M_ == 0:
                return 0
            return res / M_

        def rec(y_pred, y_true):
            M = len(y_pred)
            M_ = 0
            res = 0
            for i in range(M):
                if sum(y_true[i]) != 0:
                    res += sum(y_pred[i]*y_true[i]) / sum(y_true[i])
                    M_ += 1
            if M_ == 0:
                return 0
            return res / M_
        
        result = []
        result.append(round(mA(predictions, y_test)*100, 2))
        result.append(round(acc(predictions, y_test)*100, 2))
        prec_value = prec(predictions, y_test)
        result.append(round(prec_value*100, 2))
        rec_value = rec(predictions, y_test)
        result.append(round(rec_value*100, 2))
        result.append(round(2*prec_value*rec_value/(prec_value+rec_value)*100, 2))
        endtime = time.time()
        result.append(round(endtime - starttime, 2))
        
        return result
    
    def attributes_metrics(self):
        predictions = self.predictions 
        label = self.label
        
        selected = []
        for i in range(61):
            selected.append(i)
        selected = [[5,10,11,12],[7],[4],[18],[46]]
            
        result = []
        for i in selected:
            result.append(round(accuracy_score(predictions[:, i], label[:, i])*100, 2))
            
        return (['Carrying','Hair','Headphone','Kerchief','Collar'], result)
        #"""
        


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        type='str', default="")

    opts, args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
    if opts.gpu != "":
        ImagenetClassifier.default_args.update({'gpu_mode': True})
    else:
        ImagenetClassifier.default_args.update({'gpu_mode': False})

    # Initialize classifier + warm start by forward for allocation
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)