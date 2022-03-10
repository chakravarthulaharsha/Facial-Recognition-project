from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label  import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger


import cv2 
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

class CamApp(App):
    
    def build(self):
        self.img1 = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        self.verification_txt = Label(text="Verification Uninitiated", size_hint=(1,.1))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.img1)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_txt)

        self.model= tf.keras.models.load_model('siamesemodelv2.h5', custom_objects={'L1Dist':L1Dist})
        self.capture=cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        

        return layout
    
    def update(self, *args):
        ret, frame = self.capture.read()
        frame =frame[120:370,200:450,:]
        buf =cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr')
        self.img1.texture = img_texture


    def preprocess(file_path):
        img_byte = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img_byte)
        img = tf.image.resize(img,(100,100))
        img = img / 255.0
        return img


    def verify(self ,model, detection_threshold,verification_threshold):
        detection_threshold=0.5
        verification_threshold=0.5
        Sv_path=os.path.join('application_data','input_image','input_image.jpg')
        ret,frame=self.capture.read()
        frame = frame[120:370,200:450,:] 
        cv2.imwrite(Sv_path, frame)
        results=[]
        for image in os.listdir(os.path.join('application_data','verification_image')):
                input_img = self.preprocess(os.path.join('application_data','input_image','input_image.jpg'))
                validation_img= self.preprocess(os.path.join('application_data','verification_image',image))
        
                result= self.model.predict(list(np.expand_dims([input_img,validation_img],axis=1)))
                results.append(result)
    
        detection=np.sum(np.array(results)>detection_threshold)
        verification= detection / len(os.listdir(os.path.join('application_data','verification_image')))
        verified=verification > verification_threshold
        self.verification_txt.text ='Verified' if verification == True else 'Unverified'
        Logger.info(result)
        Logger.info(np.sum(np.array(results)>0.2))
        Logger.info(np.sum(np.array(results)>0.4))
        Logger.info(np.sum(np.array(results)>0.5))
        Logger.info(np.sum(np.array(results)>0.8))
        Logger.info(np.sum(np.array(results)>0.9))
        return results,verified

if __name__=='__main__':
    CamApp().run()