import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import time
import numpy as np

st.title('CAD SYSTEM')

@st.cache(allow_output_mutation=True)
def get_model():
        model = load_model('vggtraintop_kfold_model.h5')
        print('Model Loaded')
        return model 

from skimage.transform import resize
def run():
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
            display_img = Image.open(uploaded_file)
            st.image(display_img, use_column_width = True)
            img = np.array(Image.open(uploaded_file))
            img = resize(img, (150,150))
            img = np.reshape(img,(1,150, 150,3))
            model = get_model()
            pred = model.predict(img)
            if st.button('Predict'):
                with st.spinner('Making Prediction now...'):
                    time.sleep(3)
                    pred = np.floor(pred)
                    if pred[0][0] == 0:
                        st.write("Model tahmini {}. Sağlığınız yerinde.".format("Benign"))
                    else:
                        st.write("Model tahmini {}  Doktorunuza görünün.".format("Malign"))
                

if __name__ == '__main__':
    run()


