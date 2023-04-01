
from re import I
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
import tensorflow.keras as keras
import os
import time
from PIL import Image,ImageOps
# root='/content/drive/MyDrive/Main_Project/'
root='./'
st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:300px !important;
}
</style>
""", unsafe_allow_html=True)
localClasses=['Rice_sheath_rot', 'Leaf_smut', 'Rice_sheath_blight', 'Brown_spot', 'Bacterial_leaf_streak', 'Rice_blast', 'Leaf_scald', 'Rice_false_smut', 'Rice_stackburn', 'White_tip', 'Rice_stripe_blight', 'Rice_stem_rot']
publicClasses=['Apple___Apple_scab', 'Apple___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
           'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___healthy', 
           'Potato___Early_blight', 'Potato___healthy', 'Tomato___Early_blight', 'Tomato___healthy']
localClasses.sort()
publicClasses.sort()
prediction_label = ''

#APP

from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import MatplotlibDeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
import tensorflow as tf
from PIL import Image
normalization_layer = tf.keras.layers.experimental.preprocessing.Normalization(mean=[103.939, 116.779, 123.68],variance=[1,1,1])
import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        # print(classes[pred_index])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.8):
    # Load the original image
    # img = keras.preprocessing.image.load_img(img_path)
    # img.show()
    # img = keras.preprocessing.image.img_to_array(img)
    img=tf.convert_to_tensor(img)
    # img=tf.expand_dims(img,axis=0)
    
    img = normalization_layer(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.image.resize(superimposed_img,(224,224))
    superimposed_img = np.squeeze(superimposed_img.numpy())
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)


    # superimposed_img.save(cam_path)

    return tf.keras.utils.array_to_img(img),superimposed_img


import matplotlib.ticker as ticker
def get_concat_h(im,im1, im2,im3):
    
    dst = Image.new('RGB', (im.width+im1.width + im2.width+im3.width, im1.height))
    dst.paste(im, (0, 0))
    dst.paste(im1, (im1.width, 0))
    dst.paste(im2, (im1.width*2, 0))
    dst.paste(im3, (im1.width*3, 0))
    return dst
import io
import PIL
from PIL import Image
import matplotlib.pyplot as plt
def grad_cam_single(img_1,img_size,image,m_width,m_height,conv_with_multiplied_weight):
  # global image_path
  # image = Image.open(image_path)
  # image= image.resize(img_size,Image.BICUBIC)
 
  image1 =tf.convert_to_tensor(image)
  
  image1 = normalization_layer(image1)
  image1 = tf.cast(image1, tf.float32)*1./255
  
  cam = make_gradcam_heatmap(tf.expand_dims(image1, axis=0), model, conv_with_multiplied_weight)
  plt.figure(figsize=(3.5, 3.5))
  # plt.axis('off')
  # plt.tick_params(axis='x', which='both')
  formatter = ticker.FormatStrFormatter('%d')
  plt.tick_params(axis='both', which='both', labelsize=10)
  plt.imshow(cam,extent=[0, m_width, 0, m_height])
  tick_spacing = 1
  plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
  plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
  plt.gca().xaxis.set_major_formatter(formatter)
  plt.grid(linewidth=0)
  # img1=PIL.Image.frombytes('RGB',plt.figure.canvas.get_width_height(),plt.figure.canvas.tostring_rgb())
  img_buff=io.BytesIO()
  plt.savefig(img_buff,format='png')
  img1=Image.open(img_buff)
  # st.image(img1,width=224)

  # plt.show()
  real,superimposed_img = save_and_display_gradcam(img_1, cam,alpha=1.0)
  cam_n = Image.fromarray(np.uint8(cm.gist_earth(cam)*255))
  cam_n= cam_n.resize(img_size,Image.BICUBIC)
  img1= img1.resize(img_size,Image.BICUBIC)
  dist = get_concat_h(img1,cam_n,real,superimposed_img)
  # dist.show()
  return dist


def gradCamOutput(img1,model,image,img_size):
  conv_with_multiplied_weight = model.layers[-4].name
  m_width,m_height = model.layers[-4].output_shape[1:3]
  
  out=grad_cam_single(img1,img_size,image,m_width,m_height,conv_with_multiplied_weight)

  st.image(out,channels='RGB')
  

def load_proposed(path1,path2):
  with open(path1, "r") as json_file:
    loaded_model_json = json_file.read()
  model = tf.keras.models.model_from_json(loaded_model_json)
  model.load_weights(path2)
  return model
@st.cache_resource
def localmobileattenModel():
    #cardamom #v2s
    model_name='Mobileattenlocal'
    model=load_proposed(root+'obj_reco/local_model.json',root+'obj_reco/tst_local_model.h5')
    return model_name,model

@st.cache_resource
def publicmobileattenloadModel():
    model_name='Mobileattenpublic'
    model=load_proposed(root+'obj_reco/public_model.json',root+'obj_reco/tst_public_model.h5')
    return model_name,model


def loading():
    with st.spinner("Loading..."):
        time.sleep(1)

project = st.sidebar.radio("Dashboard",["Predictors"])

# if project == "About":
#     with st.spinner("Loading..."):
#         time.sleep(1)
#     navig = st.sidebar.radio("About",["Contributors","The APP"])
#     if navig == "Contributors":
#         st.title("Contributors of the app..")
        
#         st.header("1. Gokul R")
#         g1,g2 = st.columns([1,3])
#         g1.image("gokul_pic1.jpg")
#         g2.subheader("Gmail: gokulrajakalappan@gmail.com")
#         st.header("2. Vishnu Nandakumar")
#         v1,v2 = st.columns([1,3])
#         v1.image("vish1.jpg")
#         v2.subheader("Gmail: universalvishnu2001@gmail.com")
#         st.header("3. Gowatam rao GS")
#         gg1,gg2 = st.columns([1,3])
#         gg1.image("gow1.jpg")
#         gg2.subheader("Gmail: gowtam.rao@gmail.com")

# if project == "U^2Net_Output":
#     loading()
#     st.title("U^2 Net Output")
#     uploaded_file = st.file_uploader("Choose a image file",type=['jpg','jpeg'])
#     if uploaded_file is not None:
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         opencv_image = cv2.imdecode(file_bytes, 1)
#         image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        
#         st.write("  Uploaded Image")
#         st.image(image, channels="RGB") # Printing original image
        
#         height_temp,width_temp = image.shape[:2]
#         masked = semanticSegmenter.semanticSegmentation(image = image,apply_mask=True)
        
#         st.write("Masked Image")
#         st.image(masked, channels="RGB") # Printing 
        
#         S = semanticSegmenter.S
#         img_show = st.columns(len(S))
#         index = 0
#         for i in S:
#             i = semanticSegmenter.normPRED(i[:,0,:,:]).squeeze().cpu().data.numpy() * 255
#             i = cv2.resize(i,(width_temp,height_temp),interpolation = cv2.INTER_AREA)
#             i = i.astype(np.uint8)
#             i = cv2.resize(i,(width//2,height//2))
#             i = cv2.cvtColor(i, cv2. COLOR_GRAY2RGB)
#             img_show[index].image(i)
#             index+=1


def upload_predict(file,upload_image, model,modelsize,classes,model_name):
    
        size = modelsize  
        #image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
        #image = np.asarray(image)
        #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # img_resize = cv2.resize(upload_image, dsize=modelsize,interpolation=cv2.INTER_CUBIC)
        img_resize=upload_image
        
        st.image(img_resize, width=224)
        gradCamOutput(img_resize,model,image,modelsize)
        #img_reshape = img_resize[np.newaxis,...]
        normalization_layer = tf.keras.layers.experimental.preprocessing.Normalization(mean=[103.939, 116.779, 123.68],variance=[1,1,1])
        img=tf.convert_to_tensor(img_resize)
        img=tf.expand_dims(img,axis=0)
        img=normalization_layer(img)
        img=img*1./255
        prediction = model.predict(img)
        prediction=tf.argmax(prediction,axis=1)
        

        st.markdown('<p style="font-size:2em;">PREDICTIONS : <span style="color:red;font-size:2em;"> '+classes[prediction.numpy()[0]]+'</span></p>', unsafe_allow_html=True)
        # st.write('prediction = ',classes[prediction.numpy()[0]],model_name)
        return prediction.numpy()[0]
    

if project == "Predictors":
    loading()
    navig = st.sidebar.radio("Available Disease Predictor",["Local dataset Predictor","Public dataset Predictor"])
    predictorClasses =localClasses
    modelsize=(224,224)


    if navig == "Local dataset Predictor":
        st.title("Local dataset Predictor")
        predictorClasses = localClasses
        model_name,model = localmobileattenModel()
        modelsize=(224,224)

    if navig == "Public dataset Predictor":
        st.title("Public dataset Predictor")
        predictorClasses = publicClasses
        model_name,model = publicmobileattenloadModel()
        modelsize=(256,256)

    file = st.file_uploader("Choose a image file",type=['jpg','jpeg'])
    if file is None:
        st.text("Please upload an image file")
    else:
        #image = Image.open(file)
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        height_temp,width_temp = image.shape[:2]
        
        
        Genrate_pred,clear,bl = st.columns(3)
        gen = Genrate_pred.button("Generate Prediction")    
        # clr = clear.button("Clear",disabled=True)
        clr = False
        if gen:
            clr = clear.button("Clear",disabled=False)
            progress = st.progress(0) # intialize with 0
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i+1)
            # masked = cv2.resize(masked,(width_temp,height_temp))
            #output = samplePrediction(model,masked,predictorClasses,height,width)
            image_class = upload_predict(file.name,image, model,modelsize,predictorClasses,model_name)

        if clr:
            prediction_label = ''
